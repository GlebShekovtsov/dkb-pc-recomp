// spirv_screen_uv.c: rebuild a fragment stage's screen-position varyings from gl_FragCoord.
#pragma once
#include <stdint.h>

// A copy of the fragment module w (n words) whose NoPerspective vec4 inputs at Locations 1 (the pixel's texture
// coordinate) and 0 (the view ray scaled by c3[17].xy) are computed from gl_FragCoord instead of read; *out_n its
// words. NULL when the module does not have that layout. The caller frees the copy.
uint32_t *spirv_screen_uv(const uint32_t *w, uint64_t n, uint64_t *out_n);
