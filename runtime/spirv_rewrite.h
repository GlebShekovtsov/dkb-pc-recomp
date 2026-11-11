// SPIR-V rewrites of the translator's modules that every graphics back end applies (spirv_rewrite.c): word-level edits
// with no device and no settings behind them. Each returns a malloc'd copy the caller frees, or NULL when it has
// nothing to change (the caller then keeps the module it has). The order they run in is gfx_spirv.c's and the pipeline
// code's (vk_pipeline.c create_graphics, lib_create).
#pragma once
#include <stdint.h>

#include "gfx.h"   // GfxShaderIo

enum { SPV_OP_ENTRY_POINT = 15, SPV_OP_TYPE_INT = 21, SPV_OP_TYPE_FLOAT = 22, SPV_OP_TYPE_VECTOR = 23,
       SPV_OP_TYPE_POINTER = 32, SPV_OP_CONSTANT = 43, SPV_OP_CONSTANT_COMPOSITE = 44,
       SPV_OP_CONSTANT_NULL = 46, SPV_OP_FUNCTION = 54, SPV_OP_VARIABLE = 59,
       SPV_OP_DECORATE = 71, SPV_DEC_BUILTIN = 11, SPV_DEC_LOCATION = 30, SPV_DEC_INDEX = 32, SPV_DEC_BINDING = 33, SPV_DEC_DESCRIPTOR_SET = 34,
       SPV_SC_OUTPUT = 3 };

// Binding decorations b >= 1 become b + off.
uint32_t *spirv_rebind(const uint32_t *w, uint64_t n, uint32_t off);
// DescriptorSet decorations become set.
uint32_t *spirv_set(const uint32_t *w, uint64_t n, uint32_t set);
// The fragment Output at Location 1 moves to Location 0, Index 1 (dual-source blending). NULL without that output.
uint32_t *spirv_dual_source(const uint32_t *w, uint64_t n, uint64_t *out_n);
// NVN's alpha test before each return of the entry point: func is NVN_ALPHA_FUNC_*, ref the reference.
uint32_t *spirv_alpha_test(const uint32_t *w, uint64_t n, unsigned func, float ref, uint64_t *out_n);
// Outputs at the Locations of `missing` (typed by io) added to a vertex module and initialised to (0, 0, 0, 1).
uint32_t *spirv_add_outputs(const uint32_t *w, uint64_t n, uint32_t missing, const GfxShaderIo io[32], uint64_t *out_n);
// Unused translator declarations removed: the fragment interlock capability, and trailing ClipDistance / CullDistance
// members that put a per-vertex block over clip_cull_limit (the device's maxCombinedClipAndCullDistances; 8 when there
// is no device). geometry_chain: the vertex module feeds a geometry stage.
uint32_t *spirv_fixup_linked(const uint32_t *w, uint64_t n, uint64_t *out_n, int geometry_chain,
                             uint32_t clip_cull_limit);
uint32_t *spirv_fixup(const uint32_t *w, uint64_t n, uint64_t *out_n, uint32_t clip_cull_limit);   // geometry_chain 0
// 1 when the module declares a GroupNonUniform* or SubgroupBallot-style capability.
int spirv_uses_subgroup_ops(const uint32_t *w, uint64_t words);
// NVIDIA's fused-multiply-zero rule for float OpFMul and Fma; vertex_only leaves other stages alone.
uint32_t *spirv_fmz(const uint32_t *w, uint64_t n, int vertex_only, uint64_t *out_n, unsigned *n_mul, unsigned *n_fma);
