// SPIR-V rewrites of the translator's modules (spirv_rewrite.h): binding and set renumbering, dual-source blending,
// the alpha test, missing vertex outputs, the declaration fix-up and the fused-multiply-zero rule. They work on the
// words alone, so the Vulkan pipelines (vk_pipeline.c) and any other back end run the same code on the same module.
// Moved here from vk_pipeline.c unchanged, except that spirv_fixup_linked takes the clip and cull distance limit as an
// argument instead of asking the Vulkan device.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spirv_rewrite.h"

// Binding decorations b >= 1 become b + off (the fragment stage's share of set 0).
uint32_t *spirv_rebind(const uint32_t *w, uint64_t n, uint32_t off) {
    uint32_t *o = malloc(n * 4);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_BINDING && o[i + 3] >= 1) o[i + 3] += off;
        i += len;
    }
    return o;
}

// DescriptorSet decorations become set (the fragment stage's resources move to set 1 with independent layouts).
uint32_t *spirv_set(const uint32_t *w, uint64_t n, uint32_t set) {
    uint32_t *o = malloc(n * 4);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_DESCRIPTOR_SET) o[i + 3] = set;
        i += len;
    }
    return o;
}

// Dual-source blending (docs/maxwell.md 5.6): SRC1 blend factors read the fragment shader's colour output 1,
// which Vulkan takes from Location 0, Index 1. The Output variable at Location 1 moves there, as Ryujinx declares it
// with DualSourceBlendEnable (CodeGen/Spirv/Declarations.cs); the pipeline is keyed by GKey.dsb, so only pipelines
// that blend with SRC1 get this module. NULL when the module has no Output at Location 1.
uint32_t *spirv_dual_source(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    uint32_t outs[64];
    int nout = 0;
    uint64_t at = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        if (op == SPV_OP_VARIABLE && len >= 4 && w[i + 3] == SPV_SC_OUTPUT && nout < 64) outs[nout++] = w[i + 2];
        i += len;
    }
    for (uint64_t i = 5; i < n && !at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == SPV_OP_DECORATE && len == 4 && w[i + 2] == SPV_DEC_LOCATION && w[i + 3] == 1)
            for (int k = 0; k < nout; k++)
                if (outs[k] == w[i + 1]) at = i;
        i += len;
    }
    if (!at) return NULL;
    uint32_t *o = malloc((n + 4) * 4);
    if (!o) return NULL;
    memcpy(o, w, (at + 4) * 4);
    o[at + 3] = 0;
    o[at + 4] = 4u << 16 | SPV_OP_DECORATE, o[at + 5] = w[at + 1], o[at + 6] = SPV_DEC_INDEX, o[at + 7] = 1;
    memcpy(o + at + 8, w + at + 4, (n - at - 4) * 4);
    *out_n = n + 4;
    return o;
}

// NVN's alpha test (nvnColorStateSetAlphaTest, nvnCommandBufferSetAlphaRef) has no Vulkan state. The fragment fails when
// colour output 0's alpha does not compare true with the reference, before blending; NaN fails. So before each return of
// the entry point this adds: load Location 0, compare its w, OpKill when false. The pipeline is keyed by GKey.alpha, so
// only draws with a test get this module. NULL when the module has no float vec4 Output at Location 0.
// Two cases follow Eglantine, our default reference, without an NVN source for either. A game draw that needs the other
// answer decides them. EarlyFragmentTests stays, so with depth or stencil writes on, a rejected fragment has already
// written them. NEVER on a module without that output also gets NULL, so the draw runs with no test: Eglantine checks for
// a colour output first, Ryujinx discards first.
uint32_t *spirv_alpha_test(const uint32_t *w, uint64_t n, unsigned func, float ref, uint64_t *out_n) {
    enum { OP_TYPE_BOOL = 20, OP_FUNCTION_END = 56, OP_LOAD = 61, OP_COMPOSITE_EXTRACT = 81, OP_SELECTION_MERGE = 247,
           OP_LABEL = 248, OP_BRANCH_CONDITIONAL = 250, OP_KILL = 252, OP_RETURN = 253, RETURN_WORDS = 26 };
    // By NVN_ALPHA_FUNC_*: FOrdLessThan (a < a is never true), FOrdLessThan, FOrdEqual, FOrdLessThanEqual,
    // FOrdGreaterThan, FOrdNotEqual, FOrdGreaterThanEqual
    static const uint16_t compare[8] = {0, 184, 184, 180, 188, 186, 182, 190};
    if (func < 1 || func > 7 || n < 5) return NULL;
    uint32_t bound = w[3], entry = 0, bool_type = 0, *loc0 = calloc(bound, 4), *type_of = calloc(bound, 4);
    uint32_t *comp = calloc(bound, 4), *count = calloc(bound, 4);
    uint32_t *f32 = calloc(bound, 4), *o = NULL;
    uint32_t out_var = 0, vec = 0, flt = 0, returns = 0, bad = !loc0 || !type_of || !comp || !count || !f32;
    uint64_t fn_at = 0;
    for (uint64_t i = 5; i < n && !bad;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) { bad = 1; break; }
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && len >= 3 && !entry) {
            if (a[0] != 4) bad = 1;   // ExecutionModel Fragment
            entry = a[1];
        } else if (op == SPV_OP_DECORATE && len == 4 && a[1] == SPV_DEC_LOCATION && a[2] == 0 && a[0] < bound) loc0[a[0]] = 1;
        else if (op == OP_TYPE_BOOL && len == 2) bool_type = a[0];
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[0] < bound && a[1] == 32) f32[a[0]] = 1;
        else if (op == SPV_OP_TYPE_VECTOR && len >= 4 && a[0] < bound) comp[a[0]] = a[1], count[a[0]] = a[2];
        else if (op == SPV_OP_TYPE_POINTER && len >= 4 && a[0] < bound && a[1] == SPV_SC_OUTPUT) type_of[a[0]] = a[2];
        else if (op == SPV_OP_VARIABLE && len >= 4 && a[2] == SPV_SC_OUTPUT && a[1] < bound && loc0[a[1]] && a[0] < bound) {
            uint32_t t = type_of[a[0]];
            if (t < bound && count[t] == 4 && comp[t] < bound && f32[comp[t]]) out_var = a[1], vec = t, flt = comp[t];
        } else if (op == SPV_OP_FUNCTION) { fn_at = i; break; }
        i += len;
    }
    int in_entry = 0;
    for (uint64_t i = fn_at; fn_at && i < n && !bad;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) { bad = 1; break; }
        if (op == SPV_OP_FUNCTION) in_entry = len >= 3 && w[i + 2] == entry;
        else if (op == OP_FUNCTION_END) in_entry = 0;
        else if (op == OP_RETURN && in_entry) returns++;
        i += len;
    }
    if (!bad && out_var && entry && fn_at && returns) {
        uint32_t add = (bool_type ? 0 : 2) + 4;
        uint64_t total = n + add + (uint64_t)returns * RETURN_WORDS;
        o = malloc(total * 4);
        if (o) {
            uint32_t ref_id = bound++, bits;
            memcpy(&bits, &ref, 4);
            memcpy(o, w, fn_at * 4);
            uint64_t d = fn_at;
            if (!bool_type) bool_type = bound++, o[d++] = 2u << 16 | OP_TYPE_BOOL, o[d++] = bool_type;
            o[d++] = 4u << 16 | SPV_OP_CONSTANT, o[d++] = flt, o[d++] = ref_id, o[d++] = bits;
            in_entry = 0;
            for (uint64_t i = fn_at; i < n;) {
                uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (op == SPV_OP_FUNCTION) in_entry = w[i + 2] == entry;
                else if (op == OP_FUNCTION_END) in_entry = 0;
                else if (op == OP_RETURN && in_entry) {
                    uint32_t v = bound++, a = bound++, c = bound++, kill = bound++, merge = bound++;
                    o[d++] = 4u << 16 | OP_LOAD, o[d++] = vec, o[d++] = v, o[d++] = out_var;
                    o[d++] = 5u << 16 | OP_COMPOSITE_EXTRACT, o[d++] = flt, o[d++] = a, o[d++] = v, o[d++] = 3;
                    o[d++] = 5u << 16 | compare[func], o[d++] = bool_type, o[d++] = c, o[d++] = a;
                    o[d++] = func == 1 ? a : ref_id;
                    o[d++] = 3u << 16 | OP_SELECTION_MERGE, o[d++] = merge, o[d++] = 0;
                    o[d++] = 4u << 16 | OP_BRANCH_CONDITIONAL, o[d++] = c, o[d++] = merge, o[d++] = kill;
                    o[d++] = 2u << 16 | OP_LABEL, o[d++] = kill;
                    o[d++] = 1u << 16 | OP_KILL;
                    o[d++] = 2u << 16 | OP_LABEL, o[d++] = merge;
                }
                memcpy(o + d, w + i, len * 4);
                d += len, i += len;
            }
            o[3] = bound;
            *out_n = d;
        }
    }
    free(loc0), free(type_of), free(comp), free(count), free(f32);
    return o;
}

// Adds an Output variable for each Location of `missing` (type and width from io) to a module, declared in the entry
// point interface and initialised as Maxwell reads an unwritten output: (0, 0, 0, 1), the w as the bits of 1.0f whatever
// the type (attribute memory is untyped); fewer than 4 components are all zero. Existing scalar, vector and pointer types
// are reused. NULL when the module does not have the expected layout.
uint32_t *spirv_add_outputs(const uint32_t *w, uint64_t n, uint32_t missing, const GfxShaderIo io[32],
                                   uint64_t *out_n) {
    uint64_t ep_at = 0, ann_end = 0, fn_at = 0;
    uint32_t bound = w[3], scalar[3] = {0, 0, 0};          // GFX_TYPE_FLOAT, SINT, UINT
    struct { uint32_t id, comp, count; } vec[64];
    struct { uint32_t id, type; } ptr[64];
    int nvec = 0, nptr = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && !ep_at) ep_at = i;
        else if (op == SPV_OP_DECORATE || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633)
            ann_end = i + len;
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[1] == 32) scalar[GFX_TYPE_FLOAT] = a[0];
        else if (op == SPV_OP_TYPE_INT && len >= 4 && a[1] == 32) scalar[a[2] ? GFX_TYPE_SINT : GFX_TYPE_UINT] = a[0];
        else if (op == SPV_OP_TYPE_VECTOR && len >= 4 && nvec < 64) vec[nvec].id = a[0], vec[nvec].comp = a[1], vec[nvec++].count = a[2];
        else if (op == SPV_OP_TYPE_POINTER && len >= 4 && a[1] == SPV_SC_OUTPUT && nptr < 64) ptr[nptr].id = a[0], ptr[nptr++].type = a[2];
        else if (op == SPV_OP_FUNCTION) { fn_at = i; break; }
        i += len;
    }
    if (!ep_at || !ann_end || !fn_at || ann_end < ep_at || fn_at < ann_end) return NULL;
    uint32_t types[32 * 40], decs[32 * 4], vars[32], zero[3] = {0, 0, 0}, one[3] = {0, 0, 0};
    int nt = 0, nd = 0, nv = 0;
    for (int l = 0; l < 32; l++) {
        if (!(missing >> l & 1)) continue;
        int t = io[l].type <= GFX_TYPE_UINT ? io[l].type : GFX_TYPE_FLOAT;
        uint32_t comps = io[l].comps ? io[l].comps : 4;
        if (!scalar[t]) {
            scalar[t] = bound++;
            if (t == GFX_TYPE_FLOAT) types[nt++] = 3u << 16 | SPV_OP_TYPE_FLOAT, types[nt++] = scalar[t], types[nt++] = 32;
            else types[nt++] = 4u << 16 | SPV_OP_TYPE_INT, types[nt++] = scalar[t], types[nt++] = 32, types[nt++] = t == GFX_TYPE_SINT;
        }
        uint32_t type = scalar[t];
        if (comps > 1) {
            int k = 0;
            while (k < nvec && !(vec[k].comp == type && vec[k].count == comps)) k++;
            if (k == nvec) {
                if (nvec == 64) return NULL;
                vec[nvec].id = bound++, vec[nvec].comp = type, vec[nvec++].count = comps;
                types[nt++] = 4u << 16 | SPV_OP_TYPE_VECTOR, types[nt++] = vec[k].id, types[nt++] = type, types[nt++] = comps;
            }
            type = vec[k].id;
        }
        int k = 0;
        while (k < nptr && ptr[k].type != type) k++;
        if (k == nptr) {
            if (nptr == 64) return NULL;
            ptr[nptr].id = bound++, ptr[nptr++].type = type;
            types[nt++] = 4u << 16 | SPV_OP_TYPE_POINTER, types[nt++] = ptr[k].id, types[nt++] = SPV_SC_OUTPUT, types[nt++] = type;
        }
        uint32_t init = bound++, var = bound++;
        if (comps == 4) {
            if (!zero[t]) {
                zero[t] = bound++, one[t] = bound++;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = zero[t], types[nt++] = 0;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = one[t], types[nt++] = 0x3f800000;
            }
            types[nt++] = 7u << 16 | SPV_OP_CONSTANT_COMPOSITE, types[nt++] = type, types[nt++] = init;
            types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = one[t];
        } else {
            types[nt++] = 3u << 16 | SPV_OP_CONSTANT_NULL, types[nt++] = type, types[nt++] = init;
        }
        types[nt++] = 5u << 16 | SPV_OP_VARIABLE, types[nt++] = ptr[k].id, types[nt++] = var, types[nt++] = SPV_SC_OUTPUT,
        types[nt++] = init;
        decs[nd++] = 4u << 16 | SPV_OP_DECORATE, decs[nd++] = var, decs[nd++] = SPV_DEC_LOCATION, decs[nd++] = (uint32_t)l;
        vars[nv++] = var;
    }
    uint64_t total = n + (uint64_t)(nt + nd + nv);
    uint32_t *o = malloc(total * 4), *p = o;
    if (!o) return NULL;
    uint32_t ep_len = w[ep_at] >> 16;
    memcpy(p, w, 5 * 4), p[3] = bound, p += 5;
    memcpy(p, w + 5, (ep_at - 5) * 4), p += ep_at - 5;
    *p++ = (ep_len + (uint32_t)nv) << 16 | SPV_OP_ENTRY_POINT;
    memcpy(p, w + ep_at + 1, (ep_len - 1) * 4), p += ep_len - 1;
    memcpy(p, vars, (size_t)nv * 4), p += nv;
    memcpy(p, w + ep_at + ep_len, (ann_end - ep_at - ep_len) * 4), p += ann_end - ep_at - ep_len;
    memcpy(p, decs, (size_t)nd * 4), p += nd;
    memcpy(p, w + ann_end, (fn_at - ann_end) * 4), p += fn_at - ann_end;
    memcpy(p, types, (size_t)nt * 4), p += nt;
    memcpy(p, w + fn_at, (n - fn_at) * 4), p += n - fn_at;
    *out_n = (uint64_t)(p - o);
    return o;
}

// Remove unused translator declarations (NULL: nothing to change):
//  - the fragment shader interlock capability and extension, which Ryujinx adds to every fragment shader when the host
//    reports interlock support (SpirvGenerator.cs) and vk_device.c does not enable; kept when the shader interlocks;
//  - the CullDistance[1] member of gl_PerVertex next to ClipDistance[8] (Declarations.cs CreatePerVertexStructType):
//    9 distances exceed maxCombinedClipAndCullDistances (8 on the RX 9070). Ryujinx never accesses CullDistance; the
//    member goes when it is the struct's last and no access chain selects it;
//  - a GS has input and output per-vertex blocks. Remove their unused trailing
//    ClipDistance/CullDistance members, retaining every referenced distance and
//    every surviving member index. Declaring ClipDistance[8] in both blocks alone
//    can exceed the entry-point combined distance limit even when unused.
uint32_t *spirv_fixup_linked(const uint32_t *w, uint64_t n, uint64_t *out_n, int geometry_chain,
                             uint32_t clip_cull_limit) {
    enum { OP_EXTENSION = 10, OP_EXT_INST = 12, OP_CAPABILITY = 17, OP_MEMBER_NAME = 6, OP_EXECUTION_MODE = 16, OP_TYPE_ARRAY = 28,
           OP_TYPE_STRUCT = 30, OP_CONSTANT = 43, OP_MEMBER_DECORATE = 72, OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66,
           OP_BEGIN_INTERLOCK = 5364, OP_END_INTERLOCK = 5365, CAP_SAMPLE_INTERLOCK = 5363, CAP_PIXEL_INTERLOCK = 5378,
           CAP_RATE_INTERLOCK = 5372, BUILTIN_CLIP = 3, BUILTIN_CULL = 4 };
    if (n < 5) return NULL;
    uint32_t bound = w[3] < (1u << 22) ? w[3] : (1u << 22);
    uint32_t *val = calloc(bound, sizeof *val), *elem = calloc(bound, sizeof *elem), *len = calloc(bound, sizeof *len),
             *pointee = calloc(bound, sizeof *pointee), *var_type = calloc(bound, sizeof *var_type);
    uint8_t *constant = calloc(bound, 1);
    typedef struct { uint32_t type, cull, clip, members, member_types[8]; int used, drop; } CullFix;
    CullFix *blocks = NULL;
    uint32_t nblocks = 0, *o = NULL;
    int interlock_used = 0, interlock_declared = 0, cull_ok = 0;
    uint32_t entry_points = 0, geometry_entries = 0, vertex_entries = 0;
    if (!val || !elem || !len || !pointee || !var_type || !constant) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        if (!l || i + l > n) goto out;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && l >= 3)
            entry_points++, geometry_entries += a[0] == 3, vertex_entries += a[0] == 0;
        else if (op == OP_BEGIN_INTERLOCK || op == OP_END_INTERLOCK) interlock_used = 1;
        else if (op == OP_EXECUTION_MODE && l >= 3 && a[1] >= 5366 && a[1] <= 5371) interlock_used = 1;
        else if (op == OP_CAPABILITY && l >= 2 && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK))
            interlock_declared = 1;
        else if (op == OP_CONSTANT && l == 4 && a[1] < bound) val[a[1]] = a[2], constant[a[1]] = 1;
        else if (op == OP_TYPE_ARRAY && l >= 4 && a[0] < bound) elem[a[0]] = a[1], len[a[0]] = a[2] < bound ? val[a[2]] : 0;
        else if (op == SPV_OP_TYPE_POINTER && l >= 4 && a[0] < bound) pointee[a[0]] = a[2];
        else if (op == SPV_OP_VARIABLE && l >= 4 && a[1] < bound) var_type[a[1]] = a[0] < bound ? pointee[a[0]] : 0;
        else if (op == OP_MEMBER_DECORATE && l >= 5 && a[2] == SPV_DEC_BUILTIN &&
                 (a[3] == BUILTIN_CULL || a[3] == BUILTIN_CLIP)) {
            uint32_t j = 0;
            while (j < nblocks && blocks[j].type != a[0]) j++;
            if (j == nblocks) {
                CullFix *p = realloc(blocks, (nblocks + 1) * sizeof *p);
                if (!p) goto out;
                blocks = p;
                blocks[nblocks++] = (CullFix){.type = a[0], .cull = ~0u, .clip = ~0u};
            }
            if (a[3] == BUILTIN_CULL) blocks[j].cull = a[1];
            else blocks[j].clip = a[1];
        }
        i += l;
    }
    // A GS has separate arrayed input and output per-vertex blocks. Treat each
    // independently; used bits 1/2 retain CullDistance/ClipDistance respectively.
    int in_function = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_FUNCTION) in_function = 1;
        for (uint32_t j = 0; j < nblocks; j++) {
            CullFix *b = &blocks[j];
            if (op == OP_TYPE_STRUCT && l >= 2 && a[0] == b->type) {
                b->members = l - 2;
                for (uint32_t m = 0; m < b->members && m < 8; m++) b->member_types[m] = a[1 + m];
            } else if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && l >= 5 && a[2] < bound) {
                uint32_t t = var_type[a[2]], k = 3;
                if (t < bound && t && elem[t] == b->type) k = 4;
                if (t == b->type || k == 4) {
                    if (k + 1 >= l || a[k] >= bound || !constant[a[k]]) b->used |= 3;
                    else {
                        if (val[a[k]] == b->cull) b->used |= 1;
                        if (val[a[k]] == b->clip) b->used |= 2;
                    }
                }
            }
            if (in_function) for (uint32_t k = 0; k + 1 < l; k++) {
                // OpExtInst's fourth word is the instruction number, a literal: GLSL.std.450 Fma is 50, the same
                // number as the gl_PerVertex variable in many vertex shaders, and read as an id it kept CullDistance.
                if (op == OP_EXT_INST && k == 3) continue;
                uint32_t t = a[k] < bound ? var_type[a[k]] : 0;
                if (t != b->type && !(t < bound && t && elem[t] == b->type)) continue;
                // Only a direct, constant-member access proves which member is
                // used. Whole loads, pointer copies/calls/phis and an access to
                // the array's struct alone retain the complete block.
                if (!((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && k == 2)) b->used |= 3;
            }
        }
        i += l;
    }
    uint32_t limit = clip_cull_limit;
    for (uint32_t j = 0; j < nblocks; j++) {
        CullFix *b = &blocks[j];
        if (!b->members || b->members > 8) continue;
        if (entry_points == 1 && (geometry_entries == 1 || (geometry_chain && vertex_entries == 1))) {
            uint32_t tail = b->members;
            if (b->cull == tail - 1 && !(b->used & 1)) tail--;
            if (tail && b->clip == tail - 1 && !(b->used & 2)) tail--;
            // Keeping a nonempty block avoids needing to delete its variables
            // and entry-point interface references as well.
            if (tail) b->drop = b->members - tail;
            cull_ok |= b->drop;
            continue;
        }
        if (b->cull != b->members - 1 || b->clip >= b->members || (b->used & 1)) continue;
        uint32_t clip_t = b->member_types[b->clip], cull_t = b->member_types[b->cull];
        uint32_t clip_n = clip_t < bound ? len[clip_t] : 0, cull_n = cull_t < bound ? len[cull_t] : 0;
        b->drop = clip_n && cull_n && (uint64_t)clip_n + cull_n > limit && clip_n <= limit;
        cull_ok |= b->drop;
    }
    int strip = interlock_declared && !interlock_used;
    if (!strip && !cull_ok) goto out;
    o = malloc(n * 4);
    if (!o) goto out;
    uint64_t on = 5;
    memcpy(o, w, 5 * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        int drop = 0;
        if (strip && op == OP_CAPABILITY && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK)) drop = 1;
        if (strip && op == OP_EXTENSION && l >= 2 && !strncmp((const char *)a, "SPV_EXT_fragment_shader_interlock", (l - 1) * 4)) drop = 1;
        uint32_t shorten = 0;
        for (uint32_t j = 0; j < nblocks; j++) if (blocks[j].drop) {
            if ((op == OP_MEMBER_DECORATE || op == OP_MEMBER_NAME) && l >= 3 &&
                a[0] == blocks[j].type && a[1] >= blocks[j].members - blocks[j].drop) drop = 1;
            if (op == OP_TYPE_STRUCT && a[0] == blocks[j].type) shorten = blocks[j].drop;
        }
        if (!drop) {
            memcpy(o + on, w + i, l * 4);
            if (shorten) o[on] = (l - shorten) << 16 | op;
            on += l - shorten;
        }
        i += l;
    }
    *out_n = on;
out:
    free(val), free(elem), free(len), free(pointee), free(var_type), free(constant), free(blocks);
    return o;
}

uint32_t *spirv_fixup(const uint32_t *w, uint64_t n, uint64_t *out_n, uint32_t clip_cull_limit) {
    return spirv_fixup_linked(w, n, out_n, 0, clip_cull_limit);
}

// True when the module declares a GroupNonUniform* or SubgroupBallot-style capability (lane masks assume 32 lanes).
int spirv_uses_subgroup_ops(const uint32_t *w, uint64_t words) {
    for (uint64_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (!n || i + n > words) return 0;
        if (op != 17) return 0;   // OpCapability; capabilities come first, stop at the first other instruction
        uint32_t cap = w[i + 1];
        if ((cap >= 61 && cap <= 68) || cap == 4423 || cap == 4431) return 1;   // NonUniform*, SubgroupBallotKHR, SubgroupVoteKHR
        i += n;
    }
    return 0;
}

// DKB_FMZ=vertex|all (debug workaround for #75, off by default): NVIDIA's fused-multiply-zero rule (the DX9 rule, 0 * x = 0
// even for Inf and NaN) for 32-bit float scalar and vector OpFMul and GLSL.std.450 Fma. Each such instruction
//   r = a * b       becomes   t = a * b;     r = (a == 0 || b == 0) ? 0 : t
//   r = fma(a,b,c)  becomes   t = fma(a,b,c); r = (a == 0 || b == 0) ? c : t
// and keeps its original result id r, so decorations and uses stay valid. Only modules of the chosen stage (vertex, or
// every stage) are rewritten. NULL when nothing is rewritten.
uint32_t *spirv_fmz(const uint32_t *w, uint64_t n, int vertex_only, uint64_t *out_n, unsigned *n_mul, unsigned *n_fma) {
    enum { OP_EXT_INST_IMPORT = 11, OP_EXT_INST = 12, OP_TYPE_BOOL = 20, OP_FMUL = 133, OP_FORD_EQUAL = 180,
           OP_LOGICAL_OR = 166, OP_SELECT = 169, EXT_FMA = 50 };
    if (n < 5) return NULL;
    uint32_t bound = w[3], glsl = 0, model = ~0u, boolt[5] = {0}, nb = bound;
    uint32_t *comps = calloc(bound, 4), *f32 = calloc(bound, 4), *nullc = calloc(bound, 4), *need = calloc(bound, 4);
    uint32_t *o = NULL;
    uint32_t need_bool[5] = {0}, newbool[5] = {0};
    uint64_t fn_at = 0, extra = 0;
    unsigned mul = 0, fma = 0;
    int ok = comps && f32 && nullc && need;
    for (uint64_t i = 5; ok && i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) { ok = 0; break; }
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && len >= 3 && model == ~0u) model = a[0];
        else if (op == OP_EXT_INST_IMPORT && len >= 3 && a[0] < bound && !strcmp((const char *)(a + 1), "GLSL.std.450")) glsl = a[0];
        else if (op == OP_TYPE_BOOL && len == 2 && a[0] < bound) boolt[1] = a[0];
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[0] < bound && a[1] == 32) f32[a[0]] = 1, comps[a[0]] = 1;
        else if (op == SPV_OP_TYPE_VECTOR && len == 4 && a[0] < bound && a[1] < bound && f32[a[1]] && a[2] >= 2 && a[2] <= 4)
            f32[a[0]] = 1, comps[a[0]] = a[2];
        else if (op == SPV_OP_CONSTANT_NULL && len == 3 && a[0] < bound) nullc[a[0]] = a[1];
        else if (op == SPV_OP_FUNCTION) {
            if (!fn_at) fn_at = i;
            if (model == ~0u || (vertex_only && model != 0)) { ok = 0; break; }
        } else if ((op == OP_FMUL && len == 5) || (op == OP_EXT_INST && len == 8 && glsl && a[2] == glsl && a[3] == EXT_FMA)) {
            if (a[0] < bound && f32[a[0]]) {
                need[a[0]] = 1, need_bool[comps[a[0]]] = 1;
                if (op == OP_FMUL) mul++, extra += 21; else fma++, extra += 21;
            }
        }
        i += len;
    }
    if (!ok || !fn_at || !(mul + fma)) { free(comps), free(f32), free(nullc), free(need); return NULL; }
    for (int c = 2; c <= 4; c++) if (need_bool[c]) need_bool[1] = 1;
    for (int c = 1; c <= 4; c++) if (need_bool[c] && !boolt[c]) newbool[c] = boolt[c] = nb++, extra += c == 1 ? 2 : 4;
    for (uint32_t t = 1; t < bound; t++) if (need[t] && !nullc[t]) nullc[t] = nb++, need[t] = 2, extra += 3;
    o = malloc((n + extra + 8 * 4) * 4);
    if (!o) { free(comps), free(f32), free(nullc), free(need); return NULL; }
    uint64_t p = 0;
    memcpy(o, w, fn_at * 4), p = fn_at;
    for (int c = 1; c <= 4; c++) {
        if (!newbool[c]) continue;
        if (c == 1) o[p++] = 2u << 16 | OP_TYPE_BOOL, o[p++] = boolt[1];
        else o[p++] = 4u << 16 | SPV_OP_TYPE_VECTOR, o[p++] = boolt[c], o[p++] = boolt[1], o[p++] = c;
    }
    // OpConstantNull's result id is the one we built; the table maps type -> id, so write (type, id) pairs
    for (uint32_t t = 1; t < bound; t++)
        if (need[t] == 2) o[p++] = 3u << 16 | SPV_OP_CONSTANT_NULL, o[p++] = t, o[p++] = nullc[t];
    for (uint64_t i = fn_at; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        int is_mul = op == OP_FMUL && len == 5, is_fma = op == OP_EXT_INST && len == 8 && glsl && a[2] == glsl && a[3] == EXT_FMA;
        if ((is_mul || is_fma) && a[0] < bound && f32[a[0]]) {
            uint32_t T = a[0], r = a[1], x = a[is_mul ? 2 : 4], y = a[is_mul ? 3 : 5], B = boolt[comps[T]], N = nullc[T];
            uint32_t t = nb++, za = nb++, zb = nb++, z = nb++;
            if (is_mul) o[p++] = 5u << 16 | OP_FMUL, o[p++] = T, o[p++] = t, o[p++] = x, o[p++] = y;
            else o[p++] = 8u << 16 | OP_EXT_INST, o[p++] = T, o[p++] = t, o[p++] = glsl, o[p++] = EXT_FMA, o[p++] = x, o[p++] = y, o[p++] = a[6];
            o[p++] = 5u << 16 | OP_FORD_EQUAL, o[p++] = B, o[p++] = za, o[p++] = x, o[p++] = N;
            o[p++] = 5u << 16 | OP_FORD_EQUAL, o[p++] = B, o[p++] = zb, o[p++] = y, o[p++] = N;
            o[p++] = 5u << 16 | OP_LOGICAL_OR, o[p++] = B, o[p++] = z, o[p++] = za, o[p++] = zb;
            o[p++] = 6u << 16 | OP_SELECT, o[p++] = T, o[p++] = r, o[p++] = z, o[p++] = is_mul ? N : a[6], o[p++] = t;
        } else {
            memcpy(o + p, w + i, len * 4), p += len;
        }
        i += len;
    }
    memcpy(o, w, 5 * 4);
    o[3] = nb;
    *out_n = p;
    *n_mul = mul, *n_fma = fma;
    free(comps), free(f32), free(nullc), free(need);
    return o;
}
