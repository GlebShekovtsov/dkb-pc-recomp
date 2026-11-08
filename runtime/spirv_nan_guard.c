// NaN kept out of the colour targets (vk_pipeline.c make_module_linked, DKB_NAN_GUARD).
//
// A NaN written to a float render target does not stay one pixel: Bananza's bloom filters the HDR scene through a mip
// chain with linear sampling, every tap that touches the NaN texel turns NaN, and the composite (the fragment stage of
// the pass that adds the bloom back, reading the scene colour it writes) shows a large glowing band with black lines
// through it. Two frame dumps from an RX 9070 XT (3.2.0) have NaN in the scene colour right after the lit opaque pass:
// 5 texels on one thin diagonal line, and 15004 texels in a dithered horizontal band across a challenge room's walls
// and pillar. The game on the console shows neither, so the NaN comes from a difference in the shader path, and which
// one is not known yet.
//
// The rewrite sits at the end of that path: every OpStore to a float fragment output (Location decorated, 32-bit
// float scalar or vector, through the variable itself or an access chain into it) stores
//   isnan(v) ? marker[c] : v
// for each component c. With a marker of 0 the texel turns black instead of spreading; with a coloured marker
// (DKB_NAN_GUARD=mark) the texels show which shader wrote NaN. Integer outputs are left alone: the translator types
// each output by its render target's format, so a float output holds no integer bits.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spirv_nan_guard.h"

enum {
    OP_ENTRY_POINT = 15, OP_TYPE_BOOL = 20, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23, OP_TYPE_POINTER = 32,
    OP_CONSTANT = 43, OP_FUNCTION = 54, OP_VARIABLE = 59, OP_STORE = 62, OP_ACCESS_CHAIN = 65,
    OP_IN_BOUNDS_ACCESS_CHAIN = 66, OP_DECORATE = 71, OP_COMPOSITE_CONSTRUCT = 80, OP_COMPOSITE_EXTRACT = 81,
    OP_IS_NAN = 156, OP_SELECT = 169,
    DEC_BUILTIN = 11, DEC_LOCATION = 30, SC_OUTPUT = 3, MODEL_FRAGMENT = 4,
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

static uint32_t f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

uint32_t *spirv_nan_guard(const uint32_t *w, uint64_t n, const float marker[4], uint64_t *out_n, unsigned *n_stores) {
    *n_stores = 0;
    if (n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t bound = w[3];
    if (bound > (1u << 22)) return NULL;
    // Per id: vector component count + 1 for float32 vectors (1 for the float32 scalar), pointer storage class and
    // pointee, whether a Location / BuiltIn decorates it, and for guarded outputs and access chains into them the
    // pointee's width and the component the chain selects.
    uint8_t *fwidth = calloc(bound, 1), *has_loc = calloc(bound, 1), *builtin = calloc(bound, 1);
    uint32_t *ptr_sc = calloc(bound, 4), *ptr_to = calloc(bound, 4), *const_val = calloc(bound, 4);
    uint8_t *is_const = calloc(bound, 1), *guard = calloc(bound, 1), *chain = calloc(bound, 1);
    uint32_t *var_type = calloc(bound, 4), *chain_comp = calloc(bound, 4);
    uint32_t *out = NULL;
    Buf b = {0};
    if (!fwidth || !has_loc || !builtin || !ptr_sc || !ptr_to || !const_val || !is_const || !guard || !chain ||
        !var_type || !chain_comp)
        goto done;
    uint32_t f32 = 0, boolt = 0;
    int fragment = 0, nguard = 0;
    uint64_t fn_at = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_FUNCTION) { fn_at = i; break; }
        if (op == OP_ENTRY_POINT && len >= 3) {
            if (a[0] != MODEL_FRAGMENT) goto done;
            fragment = 1;
        } else if (op == OP_DECORATE && len >= 3 && a[0] < bound) {
            if (a[1] == DEC_LOCATION) has_loc[a[0]] = 1;
            if (a[1] == DEC_BUILTIN) builtin[a[0]] = 1;
        } else if (op == OP_TYPE_BOOL && len == 2) boolt = a[0];
        else if (op == OP_TYPE_FLOAT && len == 3 && a[1] == 32 && a[0] < bound) f32 = a[0], fwidth[a[0]] = 1;
        else if (op == OP_TYPE_VECTOR && len == 4 && a[0] < bound && f32 && a[1] == f32 && a[2] <= 4)
            fwidth[a[0]] = (uint8_t)(a[2] + 1);
        else if (op == OP_TYPE_POINTER && len == 4 && a[0] < bound) ptr_sc[a[0]] = a[1], ptr_to[a[0]] = a[2];
        else if (op == OP_CONSTANT && len == 4 && a[1] < bound) is_const[a[1]] = 1, const_val[a[1]] = a[2];
        else if (op == OP_VARIABLE && len >= 4 && a[1] < bound && a[0] < bound && a[2] == SC_OUTPUT) {
            uint32_t t = ptr_to[a[0]];
            if (t < bound && fwidth[t]) var_type[a[1]] = t;
        }
        i += len;
    }
    if (!fragment || !fn_at || !f32) goto done;
    for (uint32_t id = 1; id < bound; id++)
        if (var_type[id] && has_loc[id] && !builtin[id]) guard[id] = 1, nguard++;
    if (!nguard) goto done;

    uint32_t next = bound;
    uint32_t new_bool = boolt ? 0 : next++, use_bool = boolt ? boolt : new_bool;
    uint32_t mk[4];
    for (int c = 0; c < 4; c++) mk[c] = next++;
    put(&b, w, 5);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (i == fn_at) {   // the end of the declarations: the bool type if missing and the marker constants
            if (new_bool) PUT(&b, 2u << 16 | OP_TYPE_BOOL, new_bool);
            for (int c = 0; c < 4; c++) PUT(&b, 4u << 16 | OP_CONSTANT, f32, mk[c], f32_bits(marker[c]));
        }
        if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && len >= 5 && a[1] < bound && a[2] < bound &&
            guard[a[2]] && a[0] < bound && ptr_to[a[0]] == f32) {
            // A pointer to one float of a guarded output: its component, when the first index is a constant.
            chain[a[1]] = 1;
            chain_comp[a[1]] = a[3] < bound && is_const[a[3]] && const_val[a[3]] < 4 ? const_val[a[3]] : 0;
        } else if (op == OP_STORE && len >= 3 && a[0] < bound && (chain[a[0]] || guard[a[0]])) {
            uint32_t ptr = a[0], val = a[1];
            uint32_t width = chain[ptr] ? 1 : fwidth[var_type[ptr]] - 1;
            if (chain[ptr] || width == 1) {   // one float
                uint32_t c = chain[ptr] ? chain_comp[ptr] : 0, t = next++, s = next++;
                PUT(&b, 4u << 16 | OP_IS_NAN, use_bool, t, val);
                PUT(&b, 6u << 16 | OP_SELECT, f32, s, t, mk[c], val);
                val = s;
            } else {   // a whole vector: per component, then put back together
                uint32_t vt = var_type[ptr], parts[4], r = next++;
                for (uint32_t c = 0; c < width; c++) {
                    uint32_t e = next++, t = next++, s = next++;
                    PUT(&b, 5u << 16 | OP_COMPOSITE_EXTRACT, f32, e, val, c);
                    PUT(&b, 4u << 16 | OP_IS_NAN, use_bool, t, e);
                    PUT(&b, 6u << 16 | OP_SELECT, f32, s, t, mk[c], e);
                    parts[c] = s;
                }
                uint32_t hdr = (3 + width) << 16 | OP_COMPOSITE_CONSTRUCT, head[2] = {vt, r};
                put(&b, &hdr, 1);
                put(&b, head, 2);
                put(&b, parts, width);
                val = r;
            }
            put(&b, w + i, 2);   // OpStore and its pointer
            put(&b, &val, 1);
            put(&b, w + i + 3, len - 3);   // memory operands, if any
            (*n_stores)++;
            i += len;
            continue;
        }
        put(&b, w + i, len);
        i += len;
    }
    if (b.bad || !*n_stores) goto done;
    b.o[3] = next;
    *out_n = b.n;
    out = b.o, b.o = NULL;
done:
    free(b.o);
    free(fwidth), free(has_loc), free(builtin), free(ptr_sc), free(ptr_to), free(const_val), free(is_const);
    free(guard), free(chain), free(var_type), free(chain_comp);
    return out;
}
