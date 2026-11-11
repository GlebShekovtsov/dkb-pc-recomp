// Driver data the translated shaders read (gfx_c0.h): the NVN driver's per-stage constant bank c0 and the translator's
// support buffer. Moved here from vk_exec_draw.c (vk_c0_fill, support_fill) so that every back end fills them with the
// same code; the Vulkan replay still decides when a new copy is needed and where it lives.
#include <stdint.h>
#include <string.h>

#include "gfx_c0.h"
#include "nvn.h"

enum { NVN_STAGE_VS = NVN_SHADER_STAGE_VERTEX, NVN_STAGE_FS = NVN_SHADER_STAGE_FRAGMENT,
       NVN_STAGE_GS = NVN_SHADER_STAGE_GEOMETRY, NVN_STAGE_CS = NVN_SHADER_STAGE_COMPUTE };

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): Ryujinx's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. Other stages: zeros. Unbound slots read zero, since the replay keeps them zero.
// Layout: nxnative dm/gfx/nvn/nvn_mtl.m (MIT).
int gfx_c0_fill(uint8_t bank[GFX_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance, const Nvn2C0Layout *layout,
                const uint32_t (*roots)[NVN2_ROOT_WORDS]) {
    if (stage == NVN_STAGE_CS && layout) return nvn2_c0_pack(bank, GFX_C0_BYTES, layout, roots, ssbo, ubo);
    memset(bank, 0, GFX_C0_BYTES);
    int base =stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310 :
              stage == NVN_STAGE_GS ? 0x410 : -1;   // geometry (DKB_NATIVE_GS): the table ssbo_slot expects
    if (base < 0) return 1;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
        c0_table(bank + 0x210, ubo);
        return 1;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
    return 1;
}

void gfx_support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, GFX_SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
}
