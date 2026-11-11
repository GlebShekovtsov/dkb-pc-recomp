#pragma once
#include <stdint.h>

// A copy of fragment module w (n words) in which Bananza's HDR composite reaches the display's peak
// (spirv_hdr_shoulder.c). NULL when the module is not that composite or has a layout the rewrite does not know; the
// caller then keeps the module as it is.
uint32_t *spirv_hdr_shoulder(const uint32_t *w, uint64_t n, uint64_t *out_n);
