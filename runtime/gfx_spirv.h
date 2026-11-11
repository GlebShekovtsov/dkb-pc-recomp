// The SPIR-V a back end compiles for one shader stage (gfx_spirv.c).
#pragma once
#include <stdint.h>

#include "gfx.h"

// Module w (words) of shader src after the rewrites that depend on the stage and the settings, in this order: the
// declaration fix-up (spirv_rewrite.c spirv_fixup_linked; geometry_chain and clip_cull_limit are its arguments), the
// screen-position rebuild of the stages named in screen_uv_stage (spirv_screen_uv.c), the fused-multiply-zero rule
// (DKB_FMZ), the native-HDR highlight shoulder (spirv_hdr_shoulder.c), the NaN guard (DKB_NAN_GUARD, spirv_nan_guard.c), then the DKB_DUMP_SPV dump and the DKB_SPV_OVERRIDE
// replacement. Returns a malloc'd module the caller frees and its size in *out_words, or NULL when nothing changed
// and the caller compiles w as it is (*out_words is words then). src may be NULL: the steps that need the stage are
// skipped.
uint32_t *gfx_spirv_final(const GfxShader *src, const uint32_t *w, uint64_t words, int geometry_chain,
                          uint32_t clip_cull_limit, uint64_t *out_words);
