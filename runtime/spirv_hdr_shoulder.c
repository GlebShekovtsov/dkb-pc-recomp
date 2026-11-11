// Bananza's HDR highlights brought up to the display's peak (gfx_spirv.c gfx_spirv_final; DKB_HDR_SHOULDER=0
// turns it off).
//
// In its HDR mode (DKB_HDR_NATIVE, ov_gfx.c) the game's composite fragment stage tonemaps the scene in two steps. A
// per-channel logistic curve, (1 / (1 + exp((c1.w - x) * c1.z)) - e0) * n with e0 = 1 / (1 + exp(c1.w * c1.z)) and
// n = 1 / (1 / (1 + exp((c1.w - 1) * c1.z)) - e0), feeds the colour grade; its luminance L then goes through
// Ys = 1 - exp(-L) and, above the knee k = c[4].y, a shoulder
//   Y = k + (1 - k) * r * t / (1 + (r - 1) * (1 - t)^p),   t = (Ys - k) / (1 - k),   r = (M - k) / (1 - k)
// that runs from k to M = c[4].x, the display's peak over its paper white (p = c[4].z; c[n] is the n-th vec4 of the
// stage's user uniform block). Y scales the graded colour and M clamps it. But the logistic curve never exceeds
// (1 - e0) * n (2.54 with the c1 the game sets), so Ys never exceeds top = 1 - exp(-(1 - e0) * n) (0.921), and Y stops
// at about 1.36 times paper white whatever M is: 275 nits on a 1000-nit display (docs/hdr-native-plan.md).
//
// The rewrite runs the same shoulder from k to top instead of to 1, so the brightest input reaches M, and uses it only
// as the luminance the three colour outputs are scaled by. The chroma blend before it still reads the game's own Y,
// so hue and saturation stay as the game grades them, and below the knee nothing changes. With M at most 1 (an SDR
// variant) or top not above the knee the game's Y is kept.
//
// The stage is found by its structure, which the translator gives it: three colour output components stored as
//   FMin(FMax(0, FMul(x, Load Y)), Load U[m][4].x)
// for one Function variable Y and one uniform block U, and a comparison Load Ys >= Load U[m][4].y. The new code goes
// in front of the first of the three FMul and replaces their Y operand.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "spirv_hdr_shoulder.h"

enum {
    OP_EXT_INST_IMPORT = 11, OP_EXT_INST = 12, OP_ENTRY_POINT = 15, OP_TYPE_BOOL = 20, OP_TYPE_FLOAT = 22,
    OP_CONSTANT = 43, OP_FUNCTION = 54, OP_VARIABLE = 59, OP_LOAD = 61,
    OP_STORE = 62, OP_ACCESS_CHAIN = 65, OP_FNEGATE = 127, OP_FADD = 129, OP_FSUB = 131, OP_FMUL = 133, OP_FDIV = 136,
    OP_LOGICAL_AND = 167, OP_SELECT = 169, OP_FORD_GREATER_THAN = 186, OP_FORD_GREATER_THAN_EQUAL = 190,
    OP_FUNORD_GREATER_THAN_EQUAL = 191, OP_LABEL = 248,
    GLSL_POW = 26, GLSL_EXP = 27, GLSL_FMIN = 37, GLSL_FMAX = 40, GLSL_FCLAMP = 43,
    SC_UNIFORM = 2, SC_OUTPUT = 3, SC_FUNCTION = 7, MODEL_FRAGMENT = 4,
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

typedef struct {
    const uint32_t *w;
    uint32_t bound;
    uint64_t *def;          // per id: word offset of the instruction that defines it, 0 if none
    uint8_t *var_sc;        // per variable id: storage class + 1
    uint32_t *const_val;    // per constant id: its 32-bit value
    uint8_t *is_const;
    uint32_t glsl;          // GLSL.std.450 import
} Mod;

static const uint32_t *def_of(const Mod *m, uint32_t id, uint32_t op) {
    if (!id || id >= m->bound || !m->def[id]) return NULL;
    const uint32_t *d = m->w + m->def[id];
    return (d[0] & 0xffff) == op ? d : NULL;
}

// The pointer a Load of a Function variable reads, or 0.
static uint32_t loaded_fn_var(const Mod *m, uint32_t id) {
    const uint32_t *d = def_of(m, id, OP_LOAD);
    if (!d || (d[0] >> 16) < 4 || d[3] >= m->bound || m->var_sc[d[3]] != SC_FUNCTION + 1) return 0;
    return d[3];
}

// Whether id is a Load of U[member][row].comp for a Uniform variable U; fills *u, *member, *row_id, *comp_id and
// *ptr_type with the chain's ids when they are still 0, else requires the same U and member.
static int loads_uniform(const Mod *m, uint32_t id, uint32_t row, uint32_t comp, uint32_t *u, uint32_t *member,
                         uint32_t *row_id, uint32_t *comp_id, uint32_t *ptr_type) {
    const uint32_t *l = def_of(m, id, OP_LOAD);
    if (!l || (l[0] >> 16) < 4) return 0;
    const uint32_t *ac = def_of(m, l[3], OP_ACCESS_CHAIN);
    if (!ac || (ac[0] >> 16) != 7) return 0;
    uint32_t base = ac[3], mi = ac[4], ri = ac[5], ci = ac[6];
    if (base >= m->bound || m->var_sc[base] != SC_UNIFORM + 1) return 0;
    if (mi >= m->bound || ri >= m->bound || ci >= m->bound || !m->is_const[mi] || !m->is_const[ri] || !m->is_const[ci])
        return 0;
    if (m->const_val[ri] != row || m->const_val[ci] != comp) return 0;
    if (*u && (*u != base || m->const_val[*member] != m->const_val[mi])) return 0;
    if (!*u) *u = base, *member = mi, *ptr_type = ac[1];
    if (row_id && !*row_id) *row_id = ri;
    if (comp_id && !*comp_id) *comp_id = ci;
    return 1;
}

static int is_glsl(const Mod *m, const uint32_t *d, uint32_t inst, uint32_t nargs) {
    return d && (d[0] >> 16) == 5 + nargs && d[3] == m->glsl && d[4] == inst;
}

uint32_t *spirv_hdr_shoulder(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    if (n < 5 || w[0] != 0x07230203) return NULL;
    uint32_t bound = w[3];
    if (bound > (1u << 22)) return NULL;
    Mod m = {.w = w, .bound = bound};
    m.def = calloc(bound, sizeof *m.def);
    m.var_sc = calloc(bound, 1);
    m.const_val = calloc(bound, 4);
    m.is_const = calloc(bound, 1);
    uint32_t *out = NULL;
    Buf b = {0};
    if (!m.def || !m.var_sc || !m.const_val || !m.is_const) goto done;

    uint32_t f32 = 0, boolt = 0, out_var = 0;
    int fragment = 0;
    uint64_t fn_at = 0;
    // Definitions, the types and the import; the first output variable is the colour output (the composite has one)
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_FUNCTION && !fn_at) fn_at = i;
        if (op == OP_ENTRY_POINT && len >= 3) fragment |= a[0] == MODEL_FRAGMENT;
        else if (op == OP_EXT_INST_IMPORT && len >= 6 && a[0] < bound && !memcmp(a + 1, "GLSL.std.450", 13)) m.glsl = a[0];
        else if (op == OP_TYPE_BOOL && len == 2) boolt = a[0];
        else if (op == OP_TYPE_FLOAT && len == 3 && a[1] == 32) f32 = a[0];
        else if (op == OP_CONSTANT && len == 4 && a[1] < bound) m.is_const[a[1]] = 1, m.const_val[a[1]] = a[2];
        else if (op == OP_VARIABLE && len >= 4 && a[1] < bound) {
            m.var_sc[a[1]] = (uint8_t)(a[2] + 1);
            if (a[2] == SC_OUTPUT && !out_var) out_var = a[1];
        }
        // the result id of the value-producing instructions this rewrite follows
        if ((op == OP_LOAD || op == OP_ACCESS_CHAIN || op == OP_EXT_INST || op == OP_FMUL) && len >= 3 && a[1] < bound)
            m.def[a[1]] = i;
        i += len;
    }
    if (!fragment || !fn_at || !f32 || !boolt || !m.glsl || !out_var) goto done;

    // The three colour stores and the knee comparison
    uint32_t u = 0, member = 0, row4 = 0, comp0 = 0, ptr_type = 0, yvar = 0, ysvar = 0;
    uint64_t mul_at[3] = {0, 0, 0};
    for (uint64_t i = fn_at; i < n;) {   // u, member and the c[4] ids come from the first colour store
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto done;
        const uint32_t *a = w + i + 1;
        if (op == OP_STORE && len >= 3) {
            const uint32_t *ac = def_of(&m, a[0], OP_ACCESS_CHAIN);
            if (ac && (ac[0] >> 16) == 5 && ac[3] == out_var && ac[4] < bound && m.is_const[ac[4]] &&
                m.const_val[ac[4]] < 3) {
                uint32_t c = m.const_val[ac[4]];
                const uint32_t *fmin = def_of(&m, a[1], OP_EXT_INST);
                if (!is_glsl(&m, fmin, GLSL_FMIN, 2)) goto next;
                const uint32_t *fmax = def_of(&m, fmin[5], OP_EXT_INST);
                if (!is_glsl(&m, fmax, GLSL_FMAX, 2) ||
                    !loads_uniform(&m, fmin[6], 4, 0, &u, &member, &row4, &comp0, &ptr_type))
                    goto next;
                const uint32_t *mul = def_of(&m, fmax[6], OP_FMUL);
                // five words: the operands below are read, and the rewrite copies the whole instruction
                if (!mul || (mul[0] >> 16) != 5 || mul[1] != f32 || fmax[5] >= bound || !m.is_const[fmax[5]] ||
                    m.const_val[fmax[5]] != 0)
                    goto next;
                uint32_t v = loaded_fn_var(&m, mul[4]);
                if (!v || (yvar && v != yvar) || mul_at[c]) goto next;
                yvar = v, mul_at[c] = mul - w;
            }
        }
    next:
        i += len;
    }
    if (!mul_at[0] || !mul_at[1] || !mul_at[2]) goto done;
    uint64_t cmp_at = 0;
    for (uint64_t i = fn_at; i < n && !ysvar;) {   // Load Ys >= Load U[member][4].y
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if ((op == OP_FORD_GREATER_THAN_EQUAL || op == OP_FUNORD_GREATER_THAN_EQUAL) && len == 5 &&
            loads_uniform(&m, a[3], 4, 1, &u, &member, NULL, NULL, &ptr_type))
            ysvar = loaded_fn_var(&m, a[2]), cmp_at = i;
        i += len;
    }
    if (!ysvar || ysvar == yvar) goto done;
    uint64_t first = mul_at[0], last = mul_at[0];
    for (int c = 1; c < 3; c++) {
        if (mul_at[c] < first) first = mul_at[c];
        if (mul_at[c] > last) last = mul_at[c];
    }
    for (uint64_t i = first; i < last; i += w[i] >> 16)   // one block, so the new value is defined for all three
        if ((w[i] & 0xffff) == OP_LABEL) goto done;
    // The new code loads Ys and Y again at the first FMul. That is the Ys the knee comparison read only if nothing
    // stores to it in between, and the Y all three FMul read only if nothing stores to Y among them. Another
    // translator or game version that reuses the register would otherwise get a plausible but wrong curve; it is
    // left alone instead.
    if (cmp_at >= first) goto done;
    for (uint64_t i = cmp_at; i < last; i += w[i] >> 16)
        if ((w[i] & 0xffff) == OP_STORE && (w[i] >> 16) >= 3 &&
            (w[i + 1] == ysvar || (i >= first && w[i + 1] == yvar)))
            goto done;
    uint32_t int_t = 0;   // the index constants' type, for the new ones
    for (uint64_t i = 5; i < fn_at; i += w[i] >> 16)
        if ((w[i] & 0xffff) == OP_CONSTANT && w[i + 2] == row4) int_t = w[i + 1];
    if (!int_t) goto done;

    uint32_t next = bound;
    uint32_t k_zero = next++, k_one = next++, k_eps = next++, k_m_min = next++, k_d_min = next++;
    uint32_t i_row1 = next++, i_comp1 = next++, i_comp2 = next++, i_comp3 = next++;
    uint32_t yext = 0;
    put(&b, w, 5);
    for (uint64_t i = 5; i < n;) {
        uint32_t len = w[i] >> 16;
        if (i == fn_at) {
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_zero, f32_bits(0.0f));
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_one, f32_bits(1.0f));
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_eps, f32_bits(1e-7f));
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_m_min, f32_bits(1.001f));
            PUT(&b, 4u << 16 | OP_CONSTANT, f32, k_d_min, f32_bits(0.001f));
            PUT(&b, 4u << 16 | OP_CONSTANT, int_t, i_row1, 1);
            PUT(&b, 4u << 16 | OP_CONSTANT, int_t, i_comp1, 1);
            PUT(&b, 4u << 16 | OP_CONSTANT, int_t, i_comp2, 2);
            PUT(&b, 4u << 16 | OP_CONSTANT, int_t, i_comp3, 3);
        }
        if (i == first) {
#define LOAD_U(row, comp) ({ uint32_t ac_ = next++, ld_ = next++; \
            PUT(&b, 7u << 16 | OP_ACCESS_CHAIN, ptr_type, ac_, u, member, row, comp); \
            PUT(&b, 4u << 16 | OP_LOAD, f32, ld_, ac_); ld_; })
#define BIN(op, x, y) ({ uint32_t r_ = next++; PUT(&b, 5u << 16 | (op), f32, r_, x, y); r_; })
#define EXT1(inst, x) ({ uint32_t r_ = next++; PUT(&b, 6u << 16 | OP_EXT_INST, f32, r_, m.glsl, inst, x); r_; })
            uint32_t pk_m = LOAD_U(row4, comp0), pk_k = LOAD_U(row4, i_comp1), pk_p = LOAD_U(row4, i_comp2);
            uint32_t c1z = LOAD_U(i_row1, i_comp2), c1w = LOAD_U(i_row1, i_comp3);
            uint32_t ys = next++, y = next++;
            PUT(&b, 4u << 16 | OP_LOAD, f32, ys, ysvar);
            PUT(&b, 4u << 16 | OP_LOAD, f32, y, yvar);
            // the logistic curve's maximum (1 - e0) * n and top = 1 - exp(-that)
            uint32_t e0 = BIN(OP_FDIV, k_one, BIN(OP_FADD, k_one, EXT1(GLSL_EXP, BIN(OP_FMUL, c1w, c1z))));
            uint32_t e1 = BIN(OP_FDIV, k_one, BIN(OP_FADD, k_one,
                                                  EXT1(GLSL_EXP, BIN(OP_FMUL, BIN(OP_FSUB, c1w, k_one), c1z))));
            uint32_t span = next++;
            PUT(&b, 7u << 16 | OP_EXT_INST, f32, span, m.glsl, GLSL_FMAX, BIN(OP_FSUB, e1, e0), k_eps);
            uint32_t lmax = BIN(OP_FMUL, BIN(OP_FSUB, k_one, e0), BIN(OP_FDIV, k_one, span));
            uint32_t neg = next++;
            PUT(&b, 4u << 16 | OP_FNEGATE, f32, neg, lmax);
            uint32_t top = BIN(OP_FSUB, k_one, EXT1(GLSL_EXP, neg));
            // the shoulder from k to top
            uint32_t d = BIN(OP_FSUB, top, pk_k);
            uint32_t t = next++;
            PUT(&b, 8u << 16 | OP_EXT_INST, f32, t, m.glsl, GLSL_FCLAMP, BIN(OP_FDIV, BIN(OP_FSUB, ys, pk_k), d), k_zero,
                k_one);
            uint32_t r = BIN(OP_FDIV, BIN(OP_FSUB, pk_m, pk_k), d);
            uint32_t q = next++;
            PUT(&b, 7u << 16 | OP_EXT_INST, f32, q, m.glsl, GLSL_POW, BIN(OP_FSUB, k_one, t), pk_p);
            uint32_t den = BIN(OP_FADD, k_one, BIN(OP_FMUL, BIN(OP_FSUB, r, k_one), q));
            uint32_t ext = BIN(OP_FADD, pk_k, BIN(OP_FDIV, BIN(OP_FMUL, BIN(OP_FMUL, d, r), t), den));
            // above the knee, an HDR ceiling, and a top above the knee; else the game's Y
            uint32_t c_knee = next++, c_m = next++, c_d = next++, c_a = next++, c_all = next++;
            PUT(&b, 5u << 16 | OP_FORD_GREATER_THAN_EQUAL, boolt, c_knee, ys, pk_k);
            PUT(&b, 5u << 16 | OP_FORD_GREATER_THAN, boolt, c_m, pk_m, k_m_min);
            PUT(&b, 5u << 16 | OP_FORD_GREATER_THAN, boolt, c_d, d, k_d_min);
            PUT(&b, 5u << 16 | OP_LOGICAL_AND, boolt, c_a, c_knee, c_m);
            PUT(&b, 5u << 16 | OP_LOGICAL_AND, boolt, c_all, c_a, c_d);
            yext = next++;
            PUT(&b, 6u << 16 | OP_SELECT, f32, yext, c_all, ext, y);
#undef LOAD_U
#undef BIN
#undef EXT1
        }
        if (yext && (i == mul_at[0] || i == mul_at[1] || i == mul_at[2])) {   // FMul x, Y -> FMul x, yext
            PUT(&b, w[i], w[i + 1], w[i + 2], w[i + 3], yext);
            i += len;
            continue;
        }
        put(&b, w + i, len);
        i += len;
    }
    if (b.bad || !yext) goto done;
    b.o[3] = next;
    *out_n = b.n;
    out = b.o, b.o = NULL;
done:
    free(b.o);
    free(m.def), free(m.var_sc), free(m.const_val), free(m.is_const);
    return out;
}
