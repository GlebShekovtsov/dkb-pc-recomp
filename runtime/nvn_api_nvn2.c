// NVN 2 procs Donkey Kong Bananza asks for beyond TotK's NVN (runtime/nvn_procs_nvn2.inc). Only the calls whose effect
// the host can honour without knowing a new NVN 2 structure layout are here; the rest stop with their name
// (nvn_procs.c) until a call shows what they need.
#include <stdint.h>
#include <string.h>

#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_mode_trace.h"
#include "nvn_obj.h"

// ---- Queues. The host has one queue model: a queue's primary queue and its ray-tracing memory change nothing.
// void nvnQueueBuilderSetPrimaryQueue(NVNqueueBuilder *b, NVNqueue *primary)
NVN_API(nvnQueueBuilderSetPrimaryQueue) { NVN_LOG("QueueBuilderSetPrimaryQueue %p", arg_p(c, 1)); }
// void nvnQueueBuilderSetRayTracingMemorySize(NVNqueueBuilder *b, size_t n)
NVN_API(nvnQueueBuilderSetRayTracingMemorySize) { (void)c; }
// size_t nvnQueueBuilderGetRayTracingMemorySize(const NVNqueueBuilder *b)
NVN_API(nvnQueueBuilderGetRayTracingMemorySize) { ret_x(c, 0); }
// size_t nvnQueueGetTotalRayTracingMemoryUsed(const NVNqueue *q)
NVN_API(nvnQueueGetTotalRayTracingMemoryUsed) { ret_x(c, 0); }

// ---- Device
// int nvnDeviceGetFlags(const NVNdevice *d): the flags the device was initialized with
NVN_API(nvnDeviceGetFlags) { ret_x(c, ((const dev_t_ *)arg_p(c, 0))->flags); }
// NVNboolean nvnDeviceRegisterFastClear{Color,Colori,Colorui}Index(NVNdevice *, const T *color, NVNformat, int index),
// ...Stencil(NVNdevice *, int value): like the NVN 1 forms (nvn_api_core.c), "not registered" keeps the normal clear
NVN_API(nvnDeviceRegisterFastClearColorIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoriIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearColoruiIndex) { ret_x(c, 0); }
NVN_API(nvnDeviceRegisterFastClearStencil) { ret_x(c, 0); }

// ---- Device message buffer (driver debug messages): the host driver writes none
NVN_API(nvnDeviceMessageBufferSetDefaults) { (void)c; }
NVN_API(nvnDeviceMessageBufferSetDevice) { (void)c; }
NVN_API(nvnDeviceMessageBufferSetStorage) { (void)c; }
NVN_API(nvnDeviceMessageBufferBeginMessages) { (void)c; }
NVN_API(nvnDeviceMessageBufferEndMessages) { (void)c; }
NVN_API(nvnDeviceMessageBufferReset) { (void)c; }
// const NVNdeviceMessage *nvnDeviceMessageBufferGetNextMessage(...): no message
NVN_API(nvnDeviceMessageBufferGetNextMessage) { ret_x(c, 0); }

// ---- Multisample state: coverage truncation is a hint the Vulkan back end has no switch for
NVN_API(nvnMultisampleStateSetCoverageTruncationEnable) { (void)c; }
NVN_API(nvnMultisampleStateGetCoverageTruncationEnable) { ret_x(c, 0); }

// ---- Window colour space and HDR metadata: the host presents SDR sRGB
// void nvnWindowBuilderSetColorSpace(NVNwindowBuilder *, NVNcolorSpace), NVNcolorSpace ...GetColorSpace(...): 0, sRGB
NVN_API(nvnWindowBuilderSetColorSpace) {
    if (hdr_trace_hit(c, "nvnWindowBuilderSetColorSpace")) rt_log("hdr-trace: nvnWindowBuilderSetColorSpace x1=%d", (int)arg_x(c, 1));
    NVN_LOG("WindowBuilderSetColorSpace %d", (int)arg_x(c, 1));
}
NVN_API(nvnWindowBuilderGetColorSpace) { hdr_trace_hit(c, "nvnWindowBuilderGetColorSpace"); ret_x(c, 0); }
// void ...Set{Hdr}Display/ContentMetadata(obj, const metadata *): kept by no one
NVN_API(nvnWindowBuilderSetHdrDisplayMetadata) {
    if (hdr_trace_hit(c, "nvnWindowBuilderSetHdrDisplayMetadata")) hdr_trace_dump("nvnWindowBuilderSetHdrDisplayMetadata", arg_p(c, 1));
}
NVN_API(nvnWindowBuilderSetHdrContentMetadata) {
    if (hdr_trace_hit(c, "nvnWindowBuilderSetHdrContentMetadata")) hdr_trace_dump("nvnWindowBuilderSetHdrContentMetadata", arg_p(c, 1));
}
NVN_API(nvnWindowSetHdrDisplayMetadata) {
    if (hdr_trace_hit(c, "nvnWindowSetHdrDisplayMetadata")) hdr_trace_dump("nvnWindowSetHdrDisplayMetadata", arg_p(c, 1));
}
NVN_API(nvnWindowSetHdrContentMetadata) {
    if (hdr_trace_hit(c, "nvnWindowSetHdrContentMetadata")) hdr_trace_dump("nvnWindowSetHdrContentMetadata", arg_p(c, 1));
}

// void nvnCommandBufferSetComputeOverlapMode(NVNcommandBuffer *cmd, uint32_t mode)
// NVN 2 changes compute/graphics scheduling, not shader or resource state (docs/nvn2-command-modes.md).
// Our dispatch and graphics-pass boundaries already insert full barriers on one queue, so both scheduling modes
// retain that conservative ordering. Allowing overlap does not require the host to execute work concurrently.
NVN_API(nvnCommandBufferSetComputeOverlapMode) { (void)c; }

// ---- Command buffer calls new in NVN 2. Until the Vulkan back end learns them (mesh shaders, ray tracing, DLSS,
// variable-rate shading, conditional enables, device command buffers), they are accepted and
// recorded nowhere, and the first call of each is logged, so a run lists what the back end still has to do. None of
// them returns a value.
#define NVN2_NOT_RECORDED(n)                                                                                         \
    NVN_API(n) {                                                                                                     \
        static volatile int32_t logged;                                                                              \
        (void)c;                                                                                                     \
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log("nvn2: " #n " accepted, not recorded yet");   \
    }
// A mode call whose values up to `ok` are handled by doing nothing, with the reason in docs/nvn2-command-modes.md; a
// larger value is logged once with its number, so a scene that needs more than the host gives shows up in the log.
#define NVN2_MODE_HANDLED(n, kind, ok)                                                                                \
    NVN_API(n) {                                                                                                    \
        static volatile int32_t logged;                                                                             \
        uint32_t mode = (uint32_t)arg_x(c, 1);                                                                      \
        nvn_mode_trace(kind, arg_x(c, 0), 0, mode, 0);                                                              \
        if (mode > (ok) && !__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))                                      \
            rt_log("nvn2: " #n " mode %u is not supported (handled: 0..%u); accepted, not recorded", mode, (unsigned)(ok)); \
    }
// Recording mode: 0 and 1 are what the tutorial and cave call. 1 selects the SDK's compute synchronization path,
// which the host's barriers around dispatches and passes already cover. 2 (copy fence) to 4 (packing) are not.
NVN2_MODE_HANDLED(nvnCommandBufferSetRecordingMode, NVN_MODE_RECORDING, 1)
NVN2_NOT_RECORDED(nvnCommandBufferBeginDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferEndDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferSubmitDeviceCommands)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructures)
NVN2_NOT_RECORDED(nvnCommandBufferBindAccelerationStructuresDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferBuildAccelerationStructures)
NVN2_NOT_RECORDED(nvnCommandBufferCompactAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferCopyAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferRelocateAccelerationStructure)
NVN2_NOT_RECORDED(nvnCommandBufferDispatchRays)
NVN2_NOT_RECORDED(nvnCommandBufferDispatchRaysIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferBindDeviceMessageBuffer)
NVN2_NOT_RECORDED(nvnCommandBufferBindDeviceMessageBufferDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferResetDeviceMessageBuffer)
NVN2_NOT_RECORDED(nvnCommandBufferResetDeviceMessageBufferDeferred)
NVN2_NOT_RECORDED(nvnCommandBufferBindShadingRateImage)
NVN2_NOT_RECORDED(nvnCommandBufferSetShadingRatePaletteMode)
NVN2_NOT_RECORDED(nvnCommandBufferShadingRateImagePalette)
NVN2_NOT_RECORDED(nvnCommandBufferComputeDirtyTileMap)
NVN2_NOT_RECORDED(nvnCommandBufferDrawMeshTasks)
NVN2_NOT_RECORDED(nvnCommandBufferDrawMeshTasks3D)
NVN2_NOT_RECORDED(nvnCommandBufferDrawMeshTasks3DIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferDrawMeshTasksIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferMultiDrawMeshTasks3DIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferMultiDrawMeshTasks3DIndirectCount)
NVN2_NOT_RECORDED(nvnCommandBufferMultiDrawMeshTasksIndirect)
NVN2_NOT_RECORDED(nvnCommandBufferMultiDrawMeshTasksIndirectCount)
NVN2_NOT_RECORDED(nvnCommandBufferEnableExclusiveScissor)
NVN2_NOT_RECORDED(nvnCommandBufferSetExclusiveScissor)
NVN2_NOT_RECORDED(nvnCommandBufferSetExclusiveScissors)
NVN2_NOT_RECORDED(nvnCommandBufferInsertDebugMarkerCompute)
NVN2_NOT_RECORDED(nvnCommandBufferInsertDebugMarkerGraphics)
NVN2_NOT_RECORDED(nvnCommandBufferPushDebugGroupCompute)
NVN2_NOT_RECORDED(nvnCommandBufferPushDebugGroupGraphics)
NVN2_NOT_RECORDED(nvnCommandBufferPopDebugGroupCompute)
NVN2_NOT_RECORDED(nvnCommandBufferPopDebugGroupGraphics)
// nvnCommandBufferPush{Graphics,Compute}Constants and their Deferred forms are recorded in nvn_api_cmd.c (OP_PUSH_CONST).
NVN2_NOT_RECORDED(nvnCommandBufferSetComputeEnable)
NVN2_NOT_RECORDED(nvnCommandBufferSetComputeEnableConditional)
NVN2_NOT_RECORDED(nvnCommandBufferSetCopyEnable)
NVN2_NOT_RECORDED(nvnCommandBufferSetCopyEnableConditional)
NVN2_NOT_RECORDED(nvnCommandBufferSetConservativeRasterMode)
// Geometry processing mode 0 is the traditional vertex pipeline, the only one the replay builds (every capture so far
// called it with 0). A nonzero value selects the mesh pipeline, which has no Vulkan counterpart here yet.
NVN2_MODE_HANDLED(nvnCommandBufferSetGeometryProcessingMode, NVN_MODE_GEOMETRY, 0)
NVN2_NOT_RECORDED(nvnCommandBufferSetRepresentativeFragmentTest)

// DLSS entry points. The back end has no upscaler, so each call is accepted and answers 0 (void, or false from
// nvnDlssNetworkInitialize, so a game that checks falls back to its own path); the first call of each is logged to
// show whether the game uses DLSS at all. Keep the list in step with the nvnDlss rows of nvn_procs_nvn2.inc: a row
// missing here stops the run with "not implemented".
#define NVN2_DLSS(n)                                                                                                 \
    NVN_API(n) {                                                                                                     \
        static volatile int32_t logged;                                                                              \
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED)) rt_log("nvn2: DLSS call " #n " (accepted, not recorded)"); \
        ret_x(c, 0);                                                                                                 \
    }
NVN2_DLSS(nvnCommandBufferEvaluateDlss)
NVN2_DLSS(nvnDlssNetworkBuilderGetDevice)
NVN2_DLSS(nvnDlssNetworkBuilderGetFlags)
NVN2_DLSS(nvnDlssNetworkBuilderGetMaxEvaluationEntries)
NVN2_DLSS(nvnDlssNetworkBuilderGetMemory)
NVN2_DLSS(nvnDlssNetworkBuilderGetMemorySize)
NVN2_DLSS(nvnDlssNetworkBuilderGetMethod)
NVN2_DLSS(nvnDlssNetworkBuilderGetOptimalSettings)
NVN2_DLSS(nvnDlssNetworkBuilderGetOutputResolution)
NVN2_DLSS(nvnDlssNetworkBuilderGetPreset)
NVN2_DLSS(nvnDlssNetworkBuilderGetStorageSize)
NVN2_DLSS(nvnDlssNetworkBuilderSetDefaults)
NVN2_DLSS(nvnDlssNetworkBuilderSetDevice)
NVN2_DLSS(nvnDlssNetworkBuilderSetFlags)
NVN2_DLSS(nvnDlssNetworkBuilderSetMaxEvaluationEntries)
NVN2_DLSS(nvnDlssNetworkBuilderSetMethod)
NVN2_DLSS(nvnDlssNetworkBuilderSetOutputResolution)
NVN2_DLSS(nvnDlssNetworkBuilderSetPreset)
NVN2_DLSS(nvnDlssNetworkBuilderSetStorage)
NVN2_DLSS(nvnDlssNetworkBuilderSetTextureIDs)
NVN2_DLSS(nvnDlssNetworkFinalize)
NVN2_DLSS(nvnDlssNetworkInitialize)
NVN2_DLSS(nvnDlssParametersGetBiasCurrentColorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersGetDepthSubrectangleBase)
NVN2_DLSS(nvnDlssParametersGetEvaluationIndex)
NVN2_DLSS(nvnDlssParametersGetHistoryReset)
NVN2_DLSS(nvnDlssParametersGetInputColorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersGetInputSubrectangleSize)
NVN2_DLSS(nvnDlssParametersGetJitterOffset)
NVN2_DLSS(nvnDlssParametersGetMotionVectorScale)
NVN2_DLSS(nvnDlssParametersGetMotionVectorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersGetOutputSubrectangleBase)
NVN2_DLSS(nvnDlssParametersGetPreExposure)
NVN2_DLSS(nvnDlssParametersSetBiasCurrentColorMask)
NVN2_DLSS(nvnDlssParametersSetBiasCurrentColorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersSetDefaults)
NVN2_DLSS(nvnDlssParametersSetDepth)
NVN2_DLSS(nvnDlssParametersSetDepthSubrectangleBase)
NVN2_DLSS(nvnDlssParametersSetEvaluationIndex)
NVN2_DLSS(nvnDlssParametersSetExposureTexture)
NVN2_DLSS(nvnDlssParametersSetHistoryReset)
NVN2_DLSS(nvnDlssParametersSetInputColor)
NVN2_DLSS(nvnDlssParametersSetInputColorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersSetInputSubrectangleSize)
NVN2_DLSS(nvnDlssParametersSetJitterOffset)
NVN2_DLSS(nvnDlssParametersSetMotionVectorScale)
NVN2_DLSS(nvnDlssParametersSetMotionVectorSubrectangleBase)
NVN2_DLSS(nvnDlssParametersSetMotionVectors)
NVN2_DLSS(nvnDlssParametersSetOutputColor)
NVN2_DLSS(nvnDlssParametersSetOutputSubrectangleBase)
NVN2_DLSS(nvnDlssParametersSetOverlayDebugLayer)
NVN2_DLSS(nvnDlssParametersSetOverlayFullScreen)
NVN2_DLSS(nvnDlssParametersSetOverlayJitterDebug)
NVN2_DLSS(nvnDlssParametersSetOverlayShowNans)
NVN2_DLSS(nvnDlssParametersSetPreExposure)
