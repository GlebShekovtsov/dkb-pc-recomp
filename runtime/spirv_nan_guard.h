#pragma once
#include <stdint.h>

// A copy of fragment module w (n words) whose stores to float colour outputs replace each NaN component with
// marker[component] (spirv_nan_guard.c). NULL when the module is not a fragment stage, has no float colour output or
// has a layout the rewrite does not know; the caller then keeps the module as it is. *n_stores: stores rewritten.
uint32_t *spirv_nan_guard(const uint32_t *w, uint64_t n, const float marker[4], uint64_t *out_n, unsigned *n_stores);
