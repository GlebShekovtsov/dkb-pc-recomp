// Screen-space varyings rebuilt from the fragment's own position (vk_pipeline.c make_module_linked).
//
// Some fragment stages read their own screen position as varyings the vertex stage computed after its perspective
// divide and passed down NoPerspective. Bananza's map mask (NVN 2 fragment stage 81f30a4006e64d83, pass "R8 mask"
// before the fog fill) draws a 1020 x 900 x 1420 box the map camera sits inside, at z = w, and its vertex stage writes
//   Location 1 = (ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5)   the depth texture coordinate of the pixel
//   Location 0 = (-ndc.x * c3[17].x, -ndc.y * c3[17].y)     the view ray, scaled by the fragment's view depth
// with ndc = clip.xy / clip.w. For the faces behind the camera w < 0, and what such a vertex's divided value turns
// into depends on how the GPU clips the triangle at the near plane: NVIDIA's rasteriser gives every pixel its own
// screen position back, AMD's clipper (Eglantine shows the same on an RX 9070 XT) gives values that are not, and the
// mask comes out as long wedges along the box's faces.
//
// Both values are functions of the pixel alone, so the rewrite computes them from gl_FragCoord: the texture coordinate
// is FragCoord.xy divided by the size of the texture the stage samples (the screen-sized depth it reads at that
// coordinate), and the view ray follows from it as the vertex stage would have written it. Every OpLoad of
// components 0 and 1 of the two inputs is replaced; the shader is otherwise unchanged.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spirv_screen_uv.h"

enum {
    OP_EXT_INST_IMPORT = 11, OP_EXT_INST = 12, OP_ENTRY_POINT = 15, OP_CAPABILITY = 17, OP_TYPE_INT = 21,
    OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23, OP_TYPE_IMAGE = 25, OP_TYPE_SAMPLED_IMAGE = 27, OP_TYPE_STRUCT = 30,
    OP_TYPE_POINTER = 32, OP_CONSTANT = 43, OP_FUNCTION = 54, OP_VARIABLE = 59, OP_LOAD = 61, OP_ACCESS_CHAIN = 65,
    OP_DECORATE = 71, OP_MEMBER_DECORATE = 72, OP_COMPOSITE_EXTRACT = 81, OP_COPY_OBJECT = 83, OP_IMAGE = 100,
    OP_IMAGE_QUERY_SIZE_LOD = 103, OP_CONVERT_S_TO_F = 111, OP_FNEGATE = 127, OP_FSUB = 131, OP_FMUL = 133, OP_FDIV = 136, OP_LABEL = 248,
    DEC_BUILTIN = 11, DEC_LOCATION = 30, DEC_NO_PERSPECTIVE = 13, BUILTIN_FRAG_COORD = 15, SC_UNIFORM_CONSTANT = 0,
    SC_INPUT = 1, SC_UNIFORM = 2, CAP_IMAGE_QUERY = 50, MODEL_FRAGMENT = 4,
};

typedef struct {
    uint32_t *o;
    uint64_t n, cap;
    int bad;
} Buf;

static void put(Buf *b, const uint32_t *w, uint64_t n) {
    if (b->bad) return;
    if (b->n + n > b->cap) {
        uint64_t cap = (b->n + n) * 2 + 64;
        uint32_t *p = realloc(b->o, cap * 4);
        if (!p) { b->bad = 1; return; }
        b->o = p, b->cap = cap;
    }
    memcpy(b->o + b->n, w, n * 4);
    b->n += n;
}

#define PUT(b, ...) do { const uint32_t ins_[] = {__VA_ARGS__}; put(b, ins_, sizeof ins_ / 4); } while (0)

uint32_t *spirv_screen_uv(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    if (n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t bound = w[3];
    if (bound > (1u << 22)) return NULL;
    uint32_t *loc = malloc((size_t)bound * 4), *ptr_sc = calloc(bound, 4), *ptr_to = calloc(bound, 4);
    uint32_t *const_val = calloc(bound, 4), *const_type = calloc(bound, 4), *var_sc = calloc(bound, 4);
    uint32_t *var_type = calloc(bound, 4), *chain_of = calloc(bound, 4), *chain_comp = calloc(bound, 4);
    uint8_t *noper = calloc(bound, 1);
    uint32_t *out = NULL;
    if (!loc || !ptr_sc || !ptr_to || !const_val || !const_type || !var_sc || !var_type || !chain_of || !chain_comp ||
        !noper)
        goto done;
    memset(loc, 0xff, (size_t)bound * 4);
    uint32_t f32 = 0, i32 = 0, v2f = 0, v4f = 0, v2i = 0, entry = 0, glsl = 0;
    uint32_t sampled_var = 0, sampled_type = 0, image_type = 0, n_sampled = 0, ubo_c3 = 0, has_query = 0;
    uint64_t ep_at = 0, first_dec = 0, fn_at = 0;
    // Pass 1: the declarations.
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_FUNCTION) { fn_at = i; break; }
        if (op == OP_CAPABILITY && len == 2 && a[0] == CAP_IMAGE_QUERY) has_query = 1;
        else if (op == OP_EXT_INST_IMPORT && len >= 3 && !strcmp((const char *)(a + 1), "GLSL.std.450")) glsl = a[0];
        else if (op == OP_ENTRY_POINT && len >= 3) {
            if (entry || a[0] != MODEL_FRAGMENT) goto done;
            entry = a[1], ep_at = i;
        } else if ((op == OP_DECORATE || op == OP_MEMBER_DECORATE) && !first_dec) first_dec = i;
        if (op == OP_DECORATE && len >= 3 && a[0] < bound) {
            if (a[1] == DEC_LOCATION && len == 4) loc[a[0]] = a[2];
            if (a[1] == DEC_NO_PERSPECTIVE) noper[a[0]] = 1;
            if (a[1] == DEC_BUILTIN && len == 4 && a[2] == BUILTIN_FRAG_COORD) goto done;   // already reads it
        } else if (op == OP_TYPE_FLOAT && len == 3 && a[1] == 32) f32 = a[0];
        else if (op == OP_TYPE_INT && len == 4 && a[1] == 32 && a[2] == 1) i32 = a[0];
        else if (op == OP_TYPE_VECTOR && len == 4) {
            if (a[1] == f32 && a[2] == 2) v2f = a[0];
            if (a[1] == f32 && a[2] == 4) v4f = a[0];
            if (a[1] == i32 && i32 && a[2] == 2) v2i = a[0];
        } else if (op == OP_TYPE_SAMPLED_IMAGE && len == 3 && a[0] < bound) const_type[a[0]] = a[1] | 0x80000000u;
        else if (op == OP_TYPE_POINTER && len == 4 && a[0] < bound) ptr_sc[a[0]] = a[1], ptr_to[a[0]] = a[2];
        else if (op == OP_CONSTANT && len == 4 && a[1] < bound) const_val[a[1]] = a[2], const_type[a[1]] = a[0];
        else if (op == OP_VARIABLE && len >= 4 && a[1] < bound && a[0] < bound) {
            var_sc[a[1]] = a[2], var_type[a[1]] = ptr_to[a[0]];
            uint32_t t = ptr_to[a[0]];
            if (a[2] == SC_UNIFORM_CONSTANT && t < bound && (const_type[t] & 0x80000000u))
                n_sampled++, sampled_var = a[1], sampled_type = t, image_type = const_type[t] & 0x7fffffffu;
        }
        i += len;
    }
    if (!entry || !fn_at || !first_dec || !f32 || !i32 || !v2f || !v4f || !glsl || !has_query || n_sampled != 1)
        goto done;
    uint32_t in_uv = 0, in_ray = 0;   // the NoPerspective vec4 inputs at Locations 1 and 0
    for (uint32_t id = 1; id < bound; id++)
        if (var_sc[id] == SC_INPUT && var_type[id] == v4f && noper[id]) {
            if (loc[id] == 1) in_uv = id;
            if (loc[id] == 0) in_ray = id;
        }
    if (!in_uv || !in_ray) goto done;
    // The c3 block: the Uniform variable the shader reads as a struct of one vec4 array (Ryujinx's constant buffer
    // layout). Taken from the access chains below: the one indexed with constants 0, 14 (the depth constants).
    // Pass 2: the access chains into the two inputs and their component, and the shader's c3 variable.
    uint32_t uniform_float_ptr = 0, c0 = 0, c14 = 0;
    for (uint64_t i = fn_at; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_ACCESS_CHAIN && len == 5 && a[2] < bound && a[1] < bound && (a[2] == in_uv || a[2] == in_ray)) {
            uint32_t k = a[3];
            if (k >= bound || const_type[k] != i32 || const_val[k] > 3) goto done;
            chain_of[a[1]] = a[2], chain_comp[a[1]] = const_val[k];
        } else if (op == OP_ACCESS_CHAIN && len == 7 && a[2] < bound && var_sc[a[2]] == SC_UNIFORM &&
                   a[3] < bound && a[4] < bound && const_type[a[3]] == i32 && const_val[a[3]] == 0 &&
                   const_type[a[4]] == i32 && const_val[a[4]] == 14) {
            ubo_c3 = a[2], uniform_float_ptr = a[0], c0 = a[3], c14 = a[4];
        } else if (op == OP_LOAD && len == 4 && a[2] < bound && (a[2] == in_uv || a[2] == in_ray)) {
            goto done;   // a whole-vector read: not the layout this rewrite knows
        }
        i += len;
    }
    if (!ubo_c3 || !uniform_float_ptr || !c0 || !c14) goto done;

    // New ids: types and constants that may be missing, FragCoord and the values computed at the entry.
    uint32_t next = bound;
    uint32_t new_v2i = v2i ? 0 : next++, use_v2i = v2i ? v2i : new_v2i;
    uint32_t ptr_in_v4 = 0;
    for (uint32_t id = 1; id < bound; id++)
        if (ptr_sc[id] == SC_INPUT && ptr_to[id] == v4f) { ptr_in_v4 = id; break; }
    uint32_t new_ptr = ptr_in_v4 ? 0 : next++, use_ptr = ptr_in_v4 ? ptr_in_v4 : new_ptr;
    uint32_t frag = next++, k_int17 = next++, k_int1 = next++, k_one = next++, k_two = next++;
    uint32_t fcv = next++, fx = next++, fy = next++, si = next++, img = next++, sz = next++, szf = next++;
    uint32_t sw = next++, sh = next++, u = next++, v = next++, u2 = next++, v2 = next++, nx = next++, ny = next++;
    uint32_t kxp = next++, kx = next++, kyp = next++, ky = next++, ngx = next++, ngy = next++, rx = next++;
    uint32_t ry = next++;

    Buf b = {0};
    put(&b, w, 5);
    int in_entry = 0, entry_done = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (i == first_dec) PUT(&b, 4u << 16 | OP_DECORATE, frag, DEC_BUILTIN, BUILTIN_FRAG_COORD);
        if (i == fn_at) {   // the end of the declarations
            if (new_v2i) PUT(&b, 4u << 16 | OP_TYPE_VECTOR, new_v2i, i32, 2);
            if (new_ptr) PUT(&b, 4u << 16 | OP_TYPE_POINTER, new_ptr, SC_INPUT, v4f);
            PUT(&b, 4u << 16 | OP_VARIABLE, use_ptr, frag, SC_INPUT);
            PUT(&b, 4u << 16 | OP_CONSTANT, i32, k_int17, 17);
            PUT(&b, 4u << 16 | OP_CONSTANT, i32, k_int1, 1);
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_one, 0x3f800000u);
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_two, 0x40000000u);
        }
        if (i == ep_at) {   // the entry point lists FragCoord among its inputs
            uint32_t hdr = (len + 1) << 16 | OP_ENTRY_POINT;
            put(&b, &hdr, 1);
            put(&b, w + i + 1, len - 1);
            put(&b, &frag, 1);
            i += len;
            continue;
        }
        const uint32_t *a = w + i + 1;
        if (op == OP_LOAD && len == 4 && a[2] < bound && chain_of[a[2]] && chain_comp[a[2]] < 2) {
            uint32_t from = chain_of[a[2]] == in_uv ? (chain_comp[a[2]] ? v : u) : (chain_comp[a[2]] ? ry : rx);
            PUT(&b, 4u << 16 | OP_COPY_OBJECT, a[0], a[1], from);
            i += len;
            continue;
        }
        put(&b, w + i, len);
        if (op == OP_FUNCTION && len >= 3) in_entry = a[1] == entry;
        if (op == OP_LABEL && in_entry && !entry_done) {
            // The entry function's first block: after its OpVariables, the values every later read is replaced with.
            entry_done = 1;
            uint64_t j = i + len;
            while (j < n && (w[j] & 0xffff) == OP_VARIABLE) put(&b, w + j, w[j] >> 16), j += w[j] >> 16;
            PUT(&b, 4u << 16 | OP_LOAD, v4f, fcv, frag);
            PUT(&b, 5u << 16 | OP_COMPOSITE_EXTRACT, f32, fx, fcv, 0);
            PUT(&b, 5u << 16 | OP_COMPOSITE_EXTRACT, f32, fy, fcv, 1);
            PUT(&b, 4u << 16 | OP_LOAD, sampled_type, si, sampled_var);
            PUT(&b, 4u << 16 | OP_IMAGE, image_type, img, si);
            PUT(&b, 5u << 16 | OP_IMAGE_QUERY_SIZE_LOD, use_v2i, sz, img, c0);
            PUT(&b, 4u << 16 | OP_CONVERT_S_TO_F, v2f, szf, sz);
            PUT(&b, 5u << 16 | OP_COMPOSITE_EXTRACT, f32, sw, szf, 0);
            PUT(&b, 5u << 16 | OP_COMPOSITE_EXTRACT, f32, sh, szf, 1);
            PUT(&b, 5u << 16 | OP_FDIV, f32, u, fx, sw);   // Location 1: the pixel's texture coordinate
            PUT(&b, 5u << 16 | OP_FDIV, f32, v, fy, sh);
            PUT(&b, 5u << 16 | OP_FMUL, f32, u2, u, k_two);   // ndc.x = 2u - 1, ndc.y = 1 - 2v
            PUT(&b, 5u << 16 | OP_FSUB, f32, nx, u2, k_one);
            PUT(&b, 5u << 16 | OP_FMUL, f32, v2, v, k_two);
            PUT(&b, 5u << 16 | OP_FSUB, f32, ny, k_one, v2);
            PUT(&b, 7u << 16 | OP_ACCESS_CHAIN, uniform_float_ptr, kxp, ubo_c3, c0, k_int17, c0);
            PUT(&b, 4u << 16 | OP_LOAD, f32, kx, kxp);
            PUT(&b, 7u << 16 | OP_ACCESS_CHAIN, uniform_float_ptr, kyp, ubo_c3, c0, k_int17, k_int1);
            PUT(&b, 4u << 16 | OP_LOAD, f32, ky, kyp);
            PUT(&b, 4u << 16 | OP_FNEGATE, f32, ngx, nx);   // Location 0: (-ndc.x * K.x, -ndc.y * K.y), K = c3[17].xy
            PUT(&b, 4u << 16 | OP_FNEGATE, f32, ngy, ny);
            PUT(&b, 5u << 16 | OP_FMUL, f32, rx, ngx, kx);
            PUT(&b, 5u << 16 | OP_FMUL, f32, ry, ngy, ky);
            i = j;
            continue;
        }
        i += len;
    }
    if (b.bad) { free(b.o); goto done; }
    b.o[3] = next;
    *out_n = b.n;
    out = b.o;
done:
    free(loc), free(ptr_sc), free(ptr_to), free(const_val), free(const_type), free(var_sc), free(var_type);
    free(chain_of), free(chain_comp), free(noper);
    return out;
}
