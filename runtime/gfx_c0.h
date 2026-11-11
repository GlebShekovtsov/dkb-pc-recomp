// Driver data the translated shaders read and a graphics back end has to supply (gfx_c0.c): the NVN driver's constant
// bank c0 of one stage, and the translator's support buffer. Both are plain bytes built from replay state, so every
// back end binds the same contents.
#pragma once
#include <stdint.h>

#include "nvn2_c0.h"
#include "nvn_obj.h"

#define GFX_C0_BYTES NVN2_C0_BYTES
// c0 of one stage into bank. stage is an NVN shader stage (NVN_SHADER_STAGE_*: 0 vertex, 1 fragment, 2 geometry,
// 5 compute), ssbo and ubo its 16 bound slots, dispatch the dispatch size (compute), base_instance the draw's
// (graphics). layout: the NVN 2 compute parameter block of the program being dispatched (gfx_shader_c0_layout) with
// its root words in roots, or NULL for an NVN 1 program; it is read for the compute stage only. Returns 0 when that
// layout does not fit the bank (the bank is then unspecified), else 1.
int gfx_c0_fill(uint8_t bank[GFX_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance, const Nvn2C0Layout *layout,
                const uint32_t (*roots)[NVN2_ROOT_WORDS]);

// Ryujinx SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...
#define GFX_SUPPORT_BYTES 1408u
// The support buffer for a w x h viewport (or pass) and an NVN point size, into b[GFX_SUPPORT_BYTES].
void gfx_support_fill(uint8_t *b, float w, float h, float point_size);
