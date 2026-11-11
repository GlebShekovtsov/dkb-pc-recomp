// The SPIR-V a back end compiles for one shader stage (gfx_spirv.h): the translator's module after the fix-ups and
// workarounds that depend on the stage and on settings, and the debug dump and override of the result. Every back end
// compiles what gfx_spirv_final returns, so a workaround found on one reaches the others and a DKB_DUMP_SPV file is
// what the driver was given whichever back end ran. Moved here from vk_pipeline.c (make_module_linked) unchanged; the
// log lines keep their "vk:" prefix.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "gfx_spirv.h"
#include "host.h"
#include "platform/platform.h"
#include "spirv_hdr_shoulder.h"
#include "spirv_nan_guard.h"
#include "spirv_rewrite.h"
#include "spirv_screen_uv.h"

// DKB_DUMP_SPV=1 (debug only, for #75): the exact words handed to vkCreateShaderModule, after every rewrite, saved as
// <log folder>/spv/<vs|fs|gs|cs>-<NVN 2 stage origin>-<FNV-1a 64 of the bytes>.spv, so a SPIR-V hash from a driver
// crash dump can be matched to its NVN 2 stage. The files hold code derived from the game: local only, ignored by git.
static void spv_dump(const uint32_t *w, uint64_t words, const GfxShader *src) {
    static int on = -1;
    static volatile int64_t total;
    if (on < 0) on = cfg_get("TOTK_DUMP_SPV") != NULL;
    const char *lp = on ? rt_log_path() : NULL, *cut = lp ? strrchr(lp, '/') : NULL;
    if (!cut || !src) return;
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ ((const uint8_t *)w)[i]) * 0x100000001b3ull;
    static const char *const kind[] = {[GFX_STAGE_VS] = "vs", [GFX_STAGE_GS] = "gs", [GFX_STAGE_FS] = "fs", [GFX_STAGE_CS] = "cs"};
    int st = gfx_shader_stage(src);
    char dir[PLAT_PATH_MAX * 3], name[48], path[PLAT_PATH_MAX * 3 + 64];
    snprintf(dir, sizeof dir, "%.*s/spv", (int)(cut - lp), lp);
    snprintf(name, sizeof name, "%s-%016llx-%016llx.spv", st >= 0 && st < 6 && kind[st] ? kind[st] : "xx",
             (unsigned long long)gfx_shader_origin(src), (unsigned long long)h);
    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (plat_file_exists(path)) return;
    if (__atomic_fetch_add(&total, (int64_t)words * 4, __ATOMIC_RELAXED) + (int64_t)words * 4 > (1ll << 30)) return;   // 1 GiB cap
    plat_file_mkdir_p(dir);
    if (plat_file_write_atomic(path, w, (size_t)words * 4)) rt_log("vk: spv dump %s %llu bytes", name, (unsigned long long)words * 4);
}

// DKB_SPV_OVERRIDE=<16-hex FNV-1a 64 name>:<path.spv>[,...] (debug only, for #75, max 8 entries): a module whose final
// SPIR-V hashes to <name> (the hash in the spv_dump file name) is created from the file instead. Parsed once.
static uint32_t *spv_override(const uint32_t *w, uint64_t words, uint64_t *out_words) {
    static struct { uint64_t h; char path[PLAT_PATH_MAX * 3]; int bad; } ent[8];
    static int n = -1;
    if (n < 0) {
        n = 0;
        const char *e = cfg_get("TOTK_SPV_OVERRIDE");
        while (e && *e && n < 8) {
            char *end;
            unsigned long long h = strtoull(e, &end, 16);
            const char *p = end + 1, *q = strchr(p, ',');
            size_t len = q ? (size_t)(q - p) : strlen(p);
            if (*end == ':' && len && len < sizeof ent[0].path) {
                ent[n].h = h;
                memcpy(ent[n].path, p, len), ent[n].path[len] = 0;
                n++;
            }
            e = q ? q + 1 : NULL;
        }
    }
    if (!n) return NULL;
    uint64_t h = 0xcbf29ce484222325ull;
    for (uint64_t i = 0; i < words * 4; i++) h = (h ^ ((const uint8_t *)w)[i]) * 0x100000001b3ull;
    for (int i = 0; i < n; i++) {
        if (ent[i].h != h || ent[i].bad) continue;
        FILE *f = fopen(ent[i].path, "rb");
        long sz = -1;
        if (f && fseek(f, 0, SEEK_END) == 0) sz = ftell(f);
        uint32_t *buf = sz > 0 && sz % 4 == 0 && sz <= (16 << 20) && fseek(f, 0, SEEK_SET) == 0 ? malloc((size_t)sz) : NULL;
        if (buf && fread(buf, 1, (size_t)sz, f) == (size_t)sz) {
            fclose(f);
            rt_log("vk: SPIR-V override %016llx <- %s", (unsigned long long)h, ent[i].path);
            *out_words = (uint64_t)sz / 4;
            return buf;
        }
        if (f) fclose(f);
        free(buf);
        ent[i].bad = 1;
        rt_log("vk: SPIR-V override %016llx: cannot use %s (missing, not a multiple of 4 bytes or over 16 MiB); keeping the original",
               (unsigned long long)h, ent[i].path);
    }
    return NULL;
}

// Fragment stages whose screen-position varyings are rebuilt from gl_FragCoord (spirv_screen_uv.c): Bananza's map
// mask, which AMD's near-plane clipping turns into long wedges (Eglantine too). Named by NVN 2 stage origin, which
// gfx_shader.c keeps per input id in shader-cache/nvn2-origins.bin, so the pipelines the boot prewarm builds from the
// cache know it once the stage was lowered in any run (--translate-shaders or play). Until then the FNV-1a 64 of the
// translated SPIR-V (the "fs" hash of a TOTK_DUMP passes.txt) names it on the hosts where it was measured.
// TOTK_FIX_SCREEN_UV=0 leaves them as the game wrote them.
static int screen_uv_stage(const GfxShader *src) {
    static const uint64_t origins[] = {0x81f30a4006e64d83ull};
    static const uint64_t spirv_hashes[] = {0x544677ae4ec509beull};   // RX 9070 XT, translator d1b11706005e89ab
    static int on = -1;
    if (on < 0) on = cfg_bool("TOTK_FIX_SCREEN_UV");
    if (!on || !src || gfx_shader_stage(src) != GFX_STAGE_FS) return 0;
    uint64_t o = gfx_shader_origin(src);
    for (size_t i = 0; i < sizeof origins / sizeof *origins; i++)
        if (o == origins[i]) return 1;
    uint64_t h = gfx_shader_spirv_fnv(src);
    for (size_t i = 0; i < sizeof spirv_hashes / sizeof *spirv_hashes; i++)
        if (h == spirv_hashes[i]) return 1;
    return 0;
}

// Native HDR (spirv_hdr_shoulder.c): Bananza's HDR composite takes its highlights up to the display's peak instead of
// stopping at about 1.36 times paper white. The stage is found by its structure, so the variants with and without the
// grading LUT both get it; it runs before the NaN guard, whose selects it does not know. DKB_HDR_SHOULDER=0 keeps the
// game's curve, for a comparison or when a variant comes out wrong.
// Only with native HDR in effect (ov_gfx.c hdr_native_on: 3.2.0, hdr = 1). The game runs this stage only while it is
// told HDR is on, so a module made before a fallback to SDR is not used on the SDR swapchain.
static uint32_t *hdr_shoulder(const GfxShader *src, const uint32_t *w, uint64_t words, uint64_t *out_n) {
    static int on = -1;   // racing threads compute the same value
    if (on < 0) on = hdr_native_on() && cfg_bool("TOTK_HDR_SHOULDER");
    if (!on || !src || gfx_shader_stage(src) != GFX_STAGE_FS) return NULL;
    uint32_t *o = spirv_hdr_shoulder(w, words, out_n);
    if (o)
        rt_log("vk: fs %016llx (origin %016llx): HDR highlights extended to the display's peak",
               (unsigned long long)gfx_shader_spirv_fnv(src), (unsigned long long)gfx_shader_origin(src));
    return o;
}

// DKB_NAN_GUARD (spirv_nan_guard.c): fragment stages store 0 instead of NaN to their float colour outputs (default,
// or "zero"), a colour named by the stage instead ("mark"), or what they computed ("0"). One NaN texel in the scene
// colour is spread by the bloom into a glowing band crossed by black lines, and the console shows none. In "mark"
// mode the colour comes from the stage's SPIR-V hash (the "fs" hash of a TOTK_DUMP passes.txt): each of r, g, b is
// 0, 1, 4 or 16 from two bits of the hash, so a frame dump's _tm image tells the stage apart from the others drawn in
// the pass, and the log lists each stage with its colour.
static uint32_t *nan_guard(const GfxShader *src, const uint32_t *w, uint64_t words, uint64_t *out_n) {
    static int mode_set = -1;   // 0 off, 1 zero, 2 mark; read and written atomically
    static volatile int32_t logged;
    int mode = __atomic_load_n(&mode_set, __ATOMIC_RELAXED);
    if (mode < 0) {   // every racing thread computes the same value; only the first logs it
        const char *e = cfg_get("TOTK_NAN_GUARD");
        int m = !e || !strcmp(e, "1") || !strcmp(e, "zero") ? 1 : !strcmp(e, "0") || !*e ? 0 : !strcmp(e, "mark") ? 2 : 1;
        if (!plat_xchg32(&logged, 1)) {
            if (e && *e && strcmp(e, "0") && strcmp(e, "1") && strcmp(e, "zero") && strcmp(e, "mark"))
                rt_log("vk: DKB_NAN_GUARD=%s ignored, using zero (use 0, zero or mark)", e);
            rt_log("vk: NaN guard on fragment outputs: %s", m == 2 ? "mark" : m ? "zero" : "off");
        }
        __atomic_store_n(&mode_set, m, __ATOMIC_RELAXED);
        mode = m;
    }
    if (!mode || !src || gfx_shader_stage(src) != GFX_STAGE_FS) return NULL;
    static const float level[4] = {0.0f, 1.0f, 4.0f, 16.0f};
    uint64_t h = mode == 2 ? gfx_shader_spirv_fnv(src) : 0;
    float marker[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (mode == 2) {
        marker[0] = level[h & 3], marker[1] = level[h >> 2 & 3], marker[2] = level[h >> 4 & 3], marker[3] = 1.0f;
        if (!(h & 0x3f)) marker[0] = 16.0f;   // never black: black is what the zero mode writes
    }
    unsigned stores = 0;
    uint32_t *o = spirv_nan_guard(w, words, marker, out_n, &stores);
    if (o && mode == 2)
        rt_log("vk: fs %016llx: NaN marked as %g %g %g (%u stores)", (unsigned long long)h, marker[0], marker[1],
               marker[2], stores);
    return o;
}

uint32_t *gfx_spirv_final(const GfxShader *src, const uint32_t *w, uint64_t words, int geometry_chain,
                          uint32_t clip_cull_limit, uint64_t *out_words) {
    uint64_t fixed_n = 0;
    uint32_t *fixed = geometry_chain ? spirv_fixup_linked(w, words, &fixed_n, 1, clip_cull_limit)
                                     : spirv_fixup(w, words, &fixed_n, clip_cull_limit);
    if (fixed) w = fixed, words = fixed_n;
    uint32_t *suv = NULL;
    if (screen_uv_stage(src)) {
        uint64_t suv_n = 0;
        suv = spirv_screen_uv(w, words, &suv_n);
        if (suv) w = suv, words = suv_n;
        rt_log("vk: fs %016llx (origin %016llx): screen position %s", (unsigned long long)gfx_shader_spirv_fnv(src),
               (unsigned long long)gfx_shader_origin(src),
               suv ? "rebuilt from gl_FragCoord" : "left as written (unexpected layout)");
    }
    static int fmz_mode = -1;   // 0 off, 1 vertex stage only, 2 every stage
    if (fmz_mode < 0) {
        const char *e = cfg_get("TOTK_FMZ");
        // On by default for vertex stages (#75): the end-of-tutorial cutscene loses the NVIDIA device without it.
        fmz_mode = !e ? 1 : !strcmp(e, "0") || !*e ? 0 : !strcmp(e, "vertex") ? 1 : !strcmp(e, "all") ? 2 : 1;
        if (e && *e && strcmp(e, "0") && strcmp(e, "vertex") && strcmp(e, "all")) rt_log("vk: DKB_FMZ=%s ignored, using vertex (use 0, vertex or all)", e);
    }
    // DKB_FMZ_SEL=<mask>:<value> (hex, bisection aid for #75): only modules whose FNV-1a 64 hash before the rewrite
    // satisfies (hash & mask) == value get the FMZ rewrite, so a run can cover a fixed half, quarter, ... of them.
    static uint64_t fmz_sel_mask, fmz_sel_val;
    static int fmz_sel_read;
    if (!fmz_sel_read) {
        fmz_sel_read = 1;
        const char *s = cfg_get("TOTK_FMZ_SEL");
        char *end;
        if (s && *s) {
            fmz_sel_mask = strtoull(s, &end, 16);
            if (*end == ':') fmz_sel_val = strtoull(end + 1, NULL, 16) & fmz_sel_mask;
            else fmz_sel_mask = 0;
        }
    }
    uint32_t *fmz = NULL;
    uint64_t pre_h = 0xcbf29ce484222325ull;
    if (fmz_mode && fmz_sel_mask)
        for (uint64_t i = 0; i < words * 4; i++) pre_h = (pre_h ^ ((const uint8_t *)w)[i]) * 0x100000001b3ull;
    if (fmz_mode && (!fmz_sel_mask || (pre_h & fmz_sel_mask) == fmz_sel_val)) {
        uint64_t fmz_n = 0;
        unsigned nm = 0, nf = 0;
        fmz = spirv_fmz(w, words, fmz_mode == 1, &fmz_n, &nm, &nf);
        if (fmz) {
            w = fmz, words = fmz_n;
            uint64_t h = 0xcbf29ce484222325ull;
            for (uint64_t i = 0; i < words * 4; i++) h = (h ^ ((const uint8_t *)w)[i]) * 0x100000001b3ull;
            rt_log("vk: FMZ rewrite %016llx mul=%u fma=%u", (unsigned long long)h, nm, nf);
        }
    }
    uint64_t hs_n = 0;
    uint32_t *hs = hdr_shoulder(src, w, words, &hs_n);
    if (hs) w = hs, words = hs_n;
    uint32_t *nan = nan_guard(src, w, words, &words);
    if (nan) w = nan;
    spv_dump(w, words, src);
    uint64_t ov_n = 0;
    uint32_t *ov = spv_override(w, words, &ov_n);
    if (ov) w = ov, words = ov_n;
    // w is the last copy made, or still the caller's module: that copy is the result, the earlier ones go
    uint32_t *last = ov ? ov : nan ? nan : hs ? hs : fmz ? fmz : suv ? suv : fixed;
    if (ov != last) free(ov);
    if (nan != last) free(nan);
    if (hs != last) free(hs);
    if (fmz != last) free(fmz);
    if (suv != last) free(suv);
    if (fixed != last) free(fixed);
    *out_words = words;
    return last;
}
