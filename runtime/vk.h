// Vulkan-internal header of the graphics layer. Host code only; the guest never sees it.
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop
#include "nvn_obj.h"   // pool_t, tex_t, view_t, samp_par, NVNcopyRegion (memory and textures block)
#include "gfx.h"       // GfxShader, GfxShaderRes, GfxShaderAnswers (pipelines block)

// Required (device choice fails without them): Vulkan 1.3; VK_KHR_swapchain, VK_KHR_push_descriptor,
// VK_EXT_depth_clip_control, VK_EXT_external_memory_host; features timelineSemaphore, dynamicRendering,
// synchronization2, shaderStorageImageWriteWithoutFormat, depthClipControl, shaderInt64, samplerAnisotropy,
// independentBlend, textureCompressionBC, and geometryShader where the driver is not a portability implementation
// (Metal has no geometry stage: vk_device.c accepts the device and the pipelines that need one are skipped).
// Optional (flag set in VkCtx when enabled): the has_* members.
typedef struct VkCtx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t gfx_family;                    // graphics + compute + presentation on win32
    uint32_t xfer_family;                   // a transfer-only family when the GPU has one, else gfx_family
    VkQueue gfx_queue;                      // graphics + present
    VkQueue xfer_queue;                     // equals gfx_queue when xfer_shared
    int xfer_shared;                        // 1: no separate queue; serialise with gfx_queue
    VmaAllocator vma;
    VkSemaphore timeline;                   // one timeline semaphore; values are issued by vk_sync.c
    VkDebugUtilsMessengerEXT messenger;
    volatile int32_t validation_errors;        // errors seen by the messenger (TOTK_VK_VALIDATION=1)
    int validation;                         // layer active
    int has_custom_border_color, has_eds, has_eds2, has_eds3, has_vertex_input_dynamic_state,
        has_conditional_rendering, has_transform_feedback, has_provoking_vertex, has_device_fault, has_memory_budget,
        has_robustness2, has_buffer_device_address, has_fragment_shader_interlock, has_buffer_marker,
        has_list_restart,                   // primitiveTopologyListRestart: restart applies to list draws too (vk_exec_draw.c)
        has_nv_checkpoints,                 // VK_NV_device_diagnostic_checkpoints enabled (DKB_GPU_CHECKPOINTS=1)
        has_address_binding_report,         // VK_EXT_device_address_binding_report enabled (with the checkpoints, vk_crumbs.c)
        has_pipeline_exec_info,             // VK_KHR_pipeline_executable_properties enabled (with the checkpoints, vk_crumbs.c)
        debug_utils,                       // VK_EXT_debug_utils enabled on the instance (validation or the checkpoints)
        has_astc_ldr,                       // textureCompressionASTC_LDR (enabled when reported)
        has_hdr_metadata,                   // VK_EXT_hdr_metadata (vk_hdr.c vk_hdr_metadata)
        has_swapchain_colorspace;           // VK_EXT_swapchain_colorspace enabled on the instance (HDR10 formats)
    // Feedback loops (maxwell.md 1.9, vk_exec_draw.c feedback_prepare): a draw samples a target of its open pass that
    // an earlier draw of the pass wrote. TOTK_FEEDBACK=0: VK_FEEDBACK_OFF (sampled in GENERAL without a barrier, the
    // old behaviour); VK_FEEDBACK_LAYOUT with VK_EXT_attachment_feedback_loop_layout and dynamicRenderingLocalRead
    // (attachable images get feedback usage, a by-region barrier inside the pass); else (or TOTK_FEEDBACK=restart)
    // VK_FEEDBACK_RESTART (the pass ends and restarts behind the full barrier).
    int feedback;                           // VK_FEEDBACK_*
    uint32_t min_imported_host_alignment;   // VK_EXT_external_memory_host
    int device_index;                       // index in vkEnumeratePhysicalDevices
} VkCtx;
enum { VK_FEEDBACK_OFF, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT };

// vk_caps.c: fallbacks for GPUs other than the RX 9070, decided at device creation.
typedef struct {
    int float64;                       // shaderFloat64 enabled; 0: shaders are translated with softfloat64
    uint32_t sbuf_align;               // nonzero: storage buffers bind at their address rounded down to this
    uint32_t sg_size;                  // requiredSubgroupSize on sg_pin_stages (32)
    VkShaderStageFlags sg_pin_stages;  // 0: no pin, stages run at the device's subgroupSize
    int gpl;                           // VK_EXT_graphics_pipeline_library enabled with fast linking (TOTK_GPL=0: off)
    int amd;                           // an AMD GPU (vendor 0x1002), any driver
    int amd_sc;                        // AMD's Windows driver branch for RDNA 1 and 2 ("AMD proprietary shader compiler")
    int moltenvk;                      // the driver is MoltenVK (VK_DRIVER_ID_MOLTENVK)
    int radv;                          // Mesa's RADV (VK_DRIVER_ID_MESA_RADV)
} VkCaps;
extern VkCaps vk_caps;
void vk_caps_driver(VkPhysicalDevice p);             // first, before vkCreateDevice: vk_caps.amd, amd_sc
void vk_caps_features(VkPhysicalDeviceFeatures *f);   // before vkCreateDevice (TOTK_VK_NO_FLOAT64 clears shaderFloat64)
// Before vkCreateDevice, with the driver's graphics pipeline library feature and properties: sets vk_caps.gpl,
// 1 = enable the extensions (VK_KHR_pipeline_library, VK_EXT_graphics_pipeline_library) and the feature.
int vk_caps_gpl(const VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT *f,
                const VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT *p);
void vk_caps_init(VkCtx *c);                          // after it: pretend limits, fallbacks, the boot report

VkCtx *vk_ctx(void);                        // NULL until vk_init() succeeded
int vk_init(void);                          // idempotent; 1 on success, 0 on failure (reason logged with rt_log)
void vk_shutdown(void);
int vk_selftest(void);                      // 0 on success
// Call with the result of a failed queue submit, present or wait before stopping on it: VK_ERROR_DEVICE_LOST logs
// what VK_EXT_device_fault reports (description, faulting addresses with their precision, vendor codes) once per
// process, after the GPU breadcrumbs (vk_crumbs_report). Any other result returns at once.
void vk_device_fault_report(VkResult r);

// ---- GPU breadcrumbs (vk_crumbs.c): with VK_AMD_buffer_marker, each crumb has the GPU write its code into host memory
// twice, when the command processor reaches it and when all work before it has finished. After a lost device the two
// codes bracket the work that hung. Without the extension every call returns at once.
enum { VK_CRUMB_SUBMIT = 1, VK_CRUMB_SUBMIT_END, VK_CRUMB_UPLOAD, VK_CRUMB_UPLOAD_END, VK_CRUMB_ASTC, VK_CRUMB_PRESENT,
       VK_CRUMB_PRESENT_END, VK_CRUMB_BIND };
// TOTK_CRUMB_BIND=1 (default off, needs the buffer marker): every pipeline bind gets a VK_CRUMB_BIND crumb into words 2
// and 3 of the marker buffer. vk_crumb_bind_on says whether it is active; vk_crumb_bind_note returns the next bind
// number (the crumb's n) and remembers the pipeline and its NVN 2 stage names for the report.
int vk_crumb_bind_on(void);
uint32_t vk_crumb_bind_note(VkPipeline p, uint64_t vs, uint64_t gs, uint64_t fs);
void vk_crumbs_init(VkCtx *c);       // after the allocator exists
void vk_crumbs_release(void);
// Records crumb kind with number n (a timeline value, a batch number; the low 28 bits are kept) into cb. Only the
// thread that records cb may call it: never into a replay part while the recorder thread owns it (vk_rec.h).
void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n);
void vk_crumbs_report(void);         // logs the two codes and what they name, and the NVIDIA checkpoints
// NVIDIA: with DKB_GPU_CHECKPOINTS=1 and VK_NV_device_diagnostic_checkpoints, vk_checkpoint puts marker into cb (same
// thread rule as vk_crumb; rec_checkpoint in vk_rec.h for replay parts) and vk_checkpoint_note tells the report which
// NVN 2 stage names (vertex or compute, geometry, fragment; 0 unknown) the VkPipeline marker holds. All no-ops when off.
int vk_checkpoints_on(void);
void vk_checkpoint(VkCommandBuffer cb, const void *marker);
void vk_checkpoint_note(VkPipeline p, uint64_t vs, uint64_t gs, uint64_t fs);
// Flags for every graphics pipeline create (and library) so vk_checkpoint_note can log the driver's executables
// (statistics, ISA) the first time it sees a pipeline; 0 unless the checkpoints and VK_KHR_pipeline_executable_properties are on.
VkPipelineCreateFlags vk_pipeline_capture_flags(void);
// Marker for a non-pipeline checkpoint (rec_checkpoint): kind names the command ("dispatch", "copy buffer", ...; a
// string literal) and, for a dispatch, the compute pipeline last bound by vk_checkpoint_compute on this thread.
const void *vk_checkpoint_tag(const char *kind);
void vk_checkpoint_compute(VkPipeline p);
// With the checkpoints on and VK_EXT_device_address_binding_report: the report also maps each faulting address to the
// device-address bindings seen so far (vk_crumbs.c). Called by vk_device_fault_report for every address it prints.
void vk_crumbs_report_address(uint64_t address);

// ---- window and present (gfx_window.c, vk_present.c). Host threads only.
// vk_present_image: blits the crop rectangle {x, y, w, h} of src (NULL, or w/h <= 0: the whole w x h image), scaled
// to the swapchain extent, into the acquired swapchain image and presents it. src is in layout, which is
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL (its writes made visible to the transfer stage by the caller's barrier) or
// VK_IMAGE_LAYOUT_GENERAL (vk_texture.c images: the present makes earlier writes visible and orders later writes after
// the blit). All its writes must be submitted to gfx_queue before the call (queue order does the rest). Starts the
// window on first use. A minimised window or a lost swapchain skips the frame. Calls serialise on an internal lock.
void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// vk_present_image split in two: vk_present_blit submits the blit and leaves the present waiting, vk_present_flip
// queues it. The next blit or a swapchain rebuild queues a waiting present first.
void vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// FSR 1 on the present (vk_fsr.c): records EASU (+ RCAS) of src's crop rectangle {x, y, w, h} into the rectangle out
// {x, y, w, h} of dst, a swapchain image in TRANSFER_DST; src is in GENERAL and fmt its format. 0 = nothing recorded
// (the setting says bilinear, no upscale is needed, or FSR is not available): the caller blits.
int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                  VkImage dst, const int out[4]);
void vk_fsr_release(void);   // before the device goes away
// SMAA on the present (vk_smaa.c): with TOTK_SMAA = low|medium|high, records SMAA of the rectangle to {x, y, w, h} of the
// swapchain image dst, which is in TRANSFER_DST and holds the scaled picture, and leaves it there; 0 = nothing recorded.
int vk_smaa_record(VkCommandBuffer cb, VkImage dst, const int to[4]);
void vk_smaa_release(void);   // before the device goes away
// Frame generation (vk_framegen.c, TOTK_FRAME_GEN = 2x), called by the present thread. vk_fg_record: before the game's
// frame is presented, copies the crop of src (the window texture, in GENERAL, fmt its format, w x h) into its history and,
// when the frame before is there, makes the frame between them; returns 1 when a made frame exists (then the caller
// calls vk_fg_show in place of the blit), 0 for the plain present (off, no history, a hitch, a failure). The game's
// texture is read only by the command buffer it submits. vk_fg_show: presents the made frame, then the game's, whose
// flip waits half a frame interval, or until pending() says another job is waiting; frame (0: none) goes to
// vk_thumbs_frame for the game's frame.
int vk_fg_record(VkImage src, VkFormat fmt, int w, int h, const int crop[4]);
void vk_fg_show(int (*pending)(void), uint64_t frame);
void vk_fg_release(void);   // before the device goes away
// Colour adjustment on the present (vk_color.c): with TOTK_COLOR_* away from their defaults, records a pass that writes
// src's crop rectangle {x, y, w, h}, adjusted, into an RGBA16F image of that size in GENERAL and returns it (the caller
// scales that one instead, crop {0, 0, w, h}); VK_NULL_HANDLE = nothing recorded. src is in GENERAL, fmt its format.
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]);
void vk_color_release(void);   // before the device goes away
// HDR output (vk_hdr.c). vk_hdr_wanted: hdr, read once. vk_hdr_surface_format: the HDR10 surface format when wanted
// and offered (1). vk_hdr_mid: the RGBA8 image the present chain (blit or FSR, SMAA) writes in place of the HDR10
// swapchain image, in TRANSFER_DST_OPTIMAL; vk_hdr_record: the HDR pass from it onto the swapchain image (1; 0: nothing
// recorded). vk_hdr_ui_target / vk_hdr_ui_compose: the in-game menu's RGBA8 target, then blended onto the swapchain
// image. vk_hdr_to_srgb8: 10-bit PQ pixels to sRGB bytes.
int vk_hdr_wanted(void);
void vk_hdr_off(void);   // probes that read back 8-bit swapchain pixels, or a failed HDR setup: SDR whatever the setting
int vk_hdr_surface_format(const VkSurfaceFormatKHR *f, uint32_t n, VkSurfaceFormatKHR *out);
void vk_hdr_metadata(VkSwapchainKHR sc, int new_sc);
VkImage vk_hdr_mid(VkCommandBuffer cb, uint32_t w, uint32_t h);
int vk_hdr_record(VkCommandBuffer cb, VkImage dst, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h);
VkImageView vk_hdr_ui_target(VkCommandBuffer cb, uint32_t w, uint32_t h);
void vk_hdr_ui_compose(VkCommandBuffer cb, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h);
void vk_hdr_to_srgb8(const uint8_t *in, uint8_t *out, size_t px, int a2r);
void vk_hdr_release(void);   // before the device goes away
void vk_present_flip(void);
// TOTK_FLIGHT thumbnails (GfxBackend.thumbs_enable, thumbs_read; flight.c): the present thread blits every presented
// frame into a host-visible ring of n thumbnails.
void vk_thumbs_enable(int n);            // before the first present; n slots of FLIGHT_THUMB_W x FLIGHT_THUMB_H
void vk_thumbs_frame(uint64_t frame);    // the present thread names the frame of the vk_present_image call that follows
// Copies frame's thumbnail as RGBA when the ring still holds it complete; 0 when it was never written, is still in
// flight or was overwritten. Takes the present lock for one slot copy (about 20 us); any thread.
int vk_thumbs_read(uint64_t frame, uint8_t *rgba);
// Every submission to ctx->gfx_queue (and xfer_queue when shared) takes this lock; vk_present_image and the swapchain
// rebuild do. Recursive.
void vk_queue_lock(void);
void vk_queue_unlock(void);
// Copy of the last presented swapchain image, for probes: on, the next presents read the image back into a host buffer
// and wait for the GPU. *px stays valid until the next present; the format is the swapchain's. Returns 0 with no frame.
void vk_present_capture(int on);
int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt);

// The window system's part of device creation (gfx_window_win.c with Win32, gfx_window_sdl.c with SDL): the instance
// extensions a surface needs (their count, at most max), and whether a queue family of pd can present to it.
int gfx_window_vk_extensions(const char **out, int max);
int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family);
void gfx_vsync_bind_counter(volatile int32_t *counter);  // host-side stand-in for a bound event: incremented per tick
// The Vulkan back end's window entries (GfxBackend.window_start, window_resize, present_mode, window_hdr10, exit_wait, exit_release;
// gfx.h). gfx_window_start calls the first under its start lock: the device, window, surface and swapchain, idempotent.
void vk_present_window_start(void);
void vk_present_window_resize(void);               // the window has its new size; the swapchain follows it
const char *vk_present_mode(void);                  // the present mode the swapchain got; "" before one
int vk_present_hdr10(void);                         // the swapchain is HDR10 now (any thread)
// exit_wait: the queued present goes out, before gfx_window.c's start lock (the present thread may be waiting for it).
void vk_present_exit_wait(void);
// exit_release, under that lock: the device lets go of the window (swapchain, surface, present objects).
void vk_present_exit_release(void);
void vk_present_release(void);                     // vk_shutdown calls it first: destroys swapchain, surface, present objects
long vk_validation_errors_total(void);             // validation errors since the first vk_init, across vk_shutdown

// ---- memory and textures (vk_memory.c, vk_texture.c)
// Guest or host threads; every call needs vk_init(). The first nine have the GfxBackend signatures of gfx.h; vk_backend.c builds
// gfx_vulkan_backend from them.
int  vk_pool_init(pool_t *p);              // CPU-visible: guest storage imported (VK_EXT_external_memory_host) as one
                                           // VkBuffer; physical: device-local memory; virtual: a 64 KiB page table
void vk_pool_fini(pool_t *p);              // drops images cut from the pool; waits for the GPU if it used the pool
int  vk_pool_map_virtual(pool_t *v, int n, const void *requests);   // nvn_mapping_t[n], checked by the front end
void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);
void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s);
void vk_texture_release(const tex_t *t);   // TextureFinalize: the image stays for slots that still name its storage
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img);
void vk_guest_written(uint64_t addr, uint64_t size);   // a host write write-watch may miss: textures there re-upload

// GPU address -> backing (section 3.1). avail: bytes from offset that stay contiguous in buffer. alloc/alloc_offset:
// the device-local memory behind physical and (through the page table) virtual pool addresses, NULL for imported pools.
// cls: the pool class of the backing (VK_MCLS_*), for the traffic census.
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset, avail;
    VmaAllocation alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;   // the imported pool record behind the span (NULL for device memory), for VK_MEM_BOUND
} VkMemSpan;

// Passive snapshot only, called on the serialized replay thread (mirror state belongs to that thread). Lookup holds
// the shared pool lock but does not resolve/use/pick a span, create/sync mirrors, touch shadows or mark pools used.
// Selects the normal one-byte lookup candidate, then requires the whole request to fit that pool/contiguous backing.
// avail is capped by that original registered pool and contiguous backing. No handles/pointers or storage bytes escape.
// Success describes a currently present normal backing buffer, not later descriptor/content/write/visibility validity.
enum { VK_MEM_INSPECT_POOL_IMPORT = 1, VK_MEM_INSPECT_POOL_PHYSICAL, VK_MEM_INSPECT_POOL_VIRTUAL };
enum { VK_MEM_INSPECT_MIRROR_NONE, VK_MEM_INSPECT_MIRROR_ON, VK_MEM_INSPECT_MIRROR_REFUSED };
typedef struct {
    uint64_t offset, avail;
    uint32_t pool_kind, pool_flags;
    int32_t cls;                        // VK_MCLS_* before optional mirror picking
    uint8_t mirror_state, mirror_present, shadow_storage, buffer_present;
} VkMemInspect;
int vk_mem_inspect(uint64_t addr, uint64_t size, VkMemInspect *out);   // nonnull out: zeroed on any failure

// Canonical descriptor bytes: imported storage uses its host address (storage 0); device storage a unique physical
// lifetime and byte offset. A virtual alias resolves to the same key as its physical storage. No GPU-use side effects.
typedef struct { uint64_t storage, offset; } VkDescKey;
int vk_mem_descriptor_key(uint64_t addr, uint64_t size, VkDescKey *out);
void vk_texture_descriptors_retire(uint64_t addr, uint64_t size, uint64_t storage);
// Traffic census (vk_memory.c vk_mem_census): bytes the replay binds or copies, by pool class and use. Imported pools by
// their CPU access flag; DEVICE is physical memory (directly or through a virtual pool); SCRATCH the replay's transient
// copies. CPU_READ counts bytes host code reads from pool storage in place of the GPU (misaligned binds, byte indices,
// counter words); GPU_WRITE what copies, fills, counter reports and uniform updates write.
enum { VK_MCLS_NONE, VK_MCLS_HOST_NOACC, VK_MCLS_HOST_UNC, VK_MCLS_HOST_CACHED, VK_MCLS_DEVICE, VK_MCLS_SCRATCH, VK_MCLS_N };
enum { VK_MUSE_VB, VK_MUSE_IB, VK_MUSE_UBO, VK_MUSE_SSBO, VK_MUSE_IND, VK_MUSE_COPY_SRC, VK_MUSE_GPU_WRITE,
       VK_MUSE_CPU_READ, VK_MUSE_N };
void vk_mem_count(int cls, int use, uint64_t bytes);
// TOTK_TRACE=perf dirty census (vk_memory.c): the replay notes each imported pool it binds as a vertex, index, uniform,
// storage or indirect buffer; at the start of every submit the pages the CPU wrote in such pools since the previous
// submit are counted from the shared page generations and logged as `mem: dirty` every 60 frames. Nothing when tracing
// is off: VK_MEM_BOUND is one flag test at the call site.
void vk_mem_bound_note(void *pool);
#define VK_MEM_BOUND(sp) do { if ((rt_trace & RT_TRACE_PERF) && (sp).pool) vk_mem_bound_note((sp).pool); } while (0)
void vk_mem_dirty_submit(void);                   // vk_sync_submit, under its lock, before the replay
// VRAM mirror of buffer-bound pools (vk_memory.c). vk_mirror_on: TOTK_MEMORY=mirror|verify, or the
// device class is discrete and TOTK_MEMORY is unset or auto; set at device selection. The replay (submitting thread
// under vk_sync_submit's lock) calls:
// - VK_MEM_USE at a vertex, index, uniform, storage or indirect binding of a span vk_mem_resolve returned: the pool's
//   mirror is created on first use and sp.buffer becomes it (same offsets); a pool that stays imported keeps its buffer.
// - VK_MEM_PICK where a copy, fill or update reads or writes a span: the mirror if the pool has one, never a new one.
// - vk_mem_gpu_wrote after recording a GPU write of [sp.offset, sp.offset + size) into sp.buffer: a mirror's bytes go
//   back to guest memory at the end of the submission. No-op for imported buffers.
// - vk_mem_mirror_begin at the start of a submission's recording, vk_mem_mirror_end (records the write-backs; returns
//   the commands recorded) after its last barrier.
// vk_sync.c records the copies the mirrors' first accesses queued (vk_mem_mirror_pending) into a prologue command
// buffer with vk_mem_mirror_prologue and submits it right before each part.
extern int vk_mirror_on;
void vk_mem_use(VkMemSpan *sp);
void vk_mem_pick(VkMemSpan *sp);
#define VK_MEM_USE(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_use(&(sp)); } while (0)
#define VK_MEM_PICK(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_pick(&(sp)); } while (0)
void vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size);
uint64_t vk_mem_mirror_gen(void);   // changes whenever a mirror is made or dropped, and with every submission
int vk_mem_shadow_on(void);         // shadow storage is in use (TOTK_SHADOW)
void vk_mem_threads(void);         // the workers of sliced passes bind mirrors too (vk_exec.c), from now on
// The calling thread's vk_mem_count calls count nothing while on (the slices of a parallel pass, vk_exec.c par_record:
// the census is single-writer).
void vk_mem_count_quiet(int on);
// Uniform blocks as of the submit call (vk_usnap.c): vk_exec_record copies the uniform ranges a submission binds before
// it records, ubo bindings take the copy (0: not copied, bind memory), GPU writes of the replay drop the copies they touch.
void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence);
void vk_usnap_end(uint64_t fence);
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out);
void vk_usnap_quiet(int on);   // the calling thread's finds are not counted while on (slices of a parallel pass)
const void *vk_usnap_bytes(uint64_t addr, uint64_t n);   // the copy's bytes for the CPU; NULL: not copied or dropped
int vk_usnap_covers(uint64_t addr, uint64_t n);   // a copy of this submission overlaps the range
void vk_usnap_drop(uint64_t addr, uint64_t n);   // the replay makes the GPU write [addr, addr+n) of guest memory
uint64_t vk_usnap_arena_bytes(void);                 // arenas created so far (all kept until shutdown)
uint64_t vk_usnap_gen(void);                      // changes whenever a binding may resolve differently
void vk_usnap_shutdown(void);                     // vk_exec_draw_shutdown: frees the arenas
void vk_mem_mirror_begin(void);
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence);
int vk_mem_mirror_pending(void);
void vk_mem_mirror_prologue(VkCommandBuffer cb);
// Shadow storage (vk_memory.c): vk_sync.c calls vk_mem_shadow_sync right before each submit; the replay notes every GPU
// write into CPU-visible memory at record time (vk_mem_gpu_pending) and reports it at the fence with vk_gpu_written,
// which copies the bytes back into guest memory before telling the texture cache (vk_guest_written).
void vk_mem_shadow_sync(void);
void vk_mem_gpu_pending(uint64_t addr, uint64_t size);
void vk_gpu_written(uint64_t addr, uint64_t size);
int vk_mem_mirror_probe(uint64_t addr);   // --probe-vkexec: the pool at addr gets its mirror as on a first buffer use;
                                          // 1 when it has one
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m);   // once
int vk_mem_class_of(uint64_t addr);               // VK_MCLS_* of the pool at addr (NONE when unbacked)
void vk_mem_census(uint64_t frame);               // at each present: pool census and traffic, logged at frame 1800 and,
                                                  // with TOTK_TRACE=perf, every 1800 frames
// VRAM in use and its budget, summed over the DEVICE_LOCAL heaps. exact 1: VK_EXT_memory_budget (the whole process),
// 0: this allocator's own allocations against the heap sizes. 0 when unknown.
int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact);
// VRAM headroom (vk_headroom.c): the VMA callbacks that see new device memory, and the thread that allocates the next
// block of each memory type in use ahead of the render thread.
const VmaDeviceMemoryCallbacks *vk_headroom_callbacks(void);
void vk_headroom_start(void);
void vk_headroom_stop(void);   // before vmaDestroyAllocator
int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out);   // 1 when all of [addr, addr+size) is backed
// Changes whenever vk_mem_resolve plus VK_MEM_USE may give another span for an address: a pool added, finalized or
// mapped, a mirror created, refused or forgotten, a submission started.
uint64_t vk_mem_gen(void);
// Changes whenever a GPU address may map to other storage (vk_mem_descriptor_key): a pool added, finalized or mapped.
// Not at a submission's start and not with the mirrors, which change buffer spans only.
uint64_t vk_mem_pool_gen(void);
// Page write generations over guest memory (mem_written_pages plus vk_guest_written). Returns 1 when a page of the
// range was written after generation `since`, 0 when not, -1 when the range is not write-watched (treat as written);
// *now is the generation to pass next time.
int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);
// The same answer from per-epoch write-watch snapshots of 2 MiB regions: no system call once the regions were queried in
// this epoch (a guest submission). For checks that may see writes made during the epoch one epoch late.
int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now);
void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked);   // since the last call

// Upload batch: one command buffer on gfx_queue that texture uploads and texel copies record into. begin/end bracket
// the recording (recursive lock); vk_staging (between them) returns a mapped host buffer that lives until the batch
// completes, or, with alloc non-NULL, until the caller's vmaDestroyBuffer after vk_upload_flush(1). The replay calls
// vk_upload_flush(0) before each submission that may read uploaded images; queue order and the batch's closing
// barrier make the uploads visible to it. Returns the batch sequence number; wait=1 blocks until it completed.
// TOTK_FIX_ZERO (default 1): physical pools are zeroed once at creation and GPU-only images with memory of their own
// are cleared when created; 0 leaves both as the driver hands them out.
int vk_fix_zero(void);
VkCommandBuffer vk_upload_begin(void);
void vk_upload_end(void);
void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc);
uint64_t vk_upload_flush(int wait);
int vk_upload_done(uint64_t seq);                 // the upload batch seq (vk_upload_flush's value) has completed
// Upload batch figures since the previous call (the perf line): GPU time of the timed ones, batches submitted, time
// threads waited in vk_upload_flush(1), bytes of staging allocated.
void vk_upload_perf(uint64_t *gpu_ns, uint64_t *timed, uint64_t *batches, uint64_t *wait_ns, uint64_t *bytes);
// A timestamp query pool of `pairs` pairs on the graphics queue, or VK_NULL_HANDLE without 64-bit timestamps.
VkQueryPool vk_ts_pool(uint32_t pairs, float *period);
// Deferred destruction: freed once the upload batches recorded so far and the timeline value last noted have completed.
void vk_defer_image(VkImage img, VmaAllocation alloc);   // alloc NULL: vkDestroyImage only
void vk_defer_view(VkImageView view);
void vk_exec_image_gone(VkImage image);   // destroy_tex, texture lock held: the replay's views of image go with it
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc);
// ---- ASTC (vk_astc.c). TOTK_ASTC = auto | gpu | native | cpu | check; see vk_astc.c.
enum { VK_ASTC_CPU, VK_ASTC_GPU, VK_ASTC_NATIVE, VK_ASTC_CHECK };
int vk_astc_mode(void);                 // decided once per device (logged); VK_ASTC_CPU before vk_init
int vk_astc_native(VkFormat astc_fmt);  // 1: images of this ASTC format are created as such and get the raw blocks
int vk_astc_gpu(void);                  // 1: ASTC images decoded to RGBA8 by the compute decoder (vk_astc_record)
int vk_astc_bc(void);                   // 1: those images are BC3 instead, encoded after the decode (TOTK_ASTC_BC=1)
// One image region decoded from raw blocks: tight rows of 16-byte blocks at byte src of the source buffer (slices one
// after the other), w x h texels and slices deep, written at offset of level / layer.
typedef struct { VkDeviceSize src; uint32_t w, h, slices, level, layer; VkOffset3D offset; } VkAstcRegion;
// Records into cb, the open upload batch (between vk_upload_begin and vk_upload_end): src (TRANSFER_SRC, src_size
// bytes of blocks) goes to a device-local scratch buffer, a dispatch per region decodes it, and the texels are copied
// into img (VK_IMAGE_LAYOUT_GENERAL). The caller records its usual barriers around an image copy before and after the
// call. readback (optional, host-visible, TRANSFER_DST): also gets the decoded texels, the regions one after the other.
// bc: img is BC3 (TexVkFormat.astc_bc); the texels are encoded to BC3 blocks before the copy, and readback is ignored.
// 0 when the decoder is unavailable or the scratch buffer cannot be allocated (nothing recorded).
int vk_astc_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_size, const VkAstcRegion *rg, int n, uint32_t bw,
                   uint32_t bh, int srgb, int bc, VkImage img, VkImageAspectFlags aspect, VkBuffer readback);
void vk_astc_stats(uint64_t *dispatches, uint64_t *texels);   // since start
// TOTK_ASTC_TIME=1: GPU timestamps around each vk_astc_record's dispatches (logged per footprint and the largest one)
void vk_astc_time_enable(int on);
int vk_astc_time_get(uint32_t bw, uint32_t bh, uint64_t *texels, uint64_t *calls, double *us);   // totals so far
void vk_astc_time_summary(void);        // logs the totals
void vk_astc_release(void);             // vk_device.c, before the device goes: frees the decoder's objects
// Protect objects retired from here on until this timeline value completes. Announce before recording their users;
// this is a destruction frontier, not a report that the value was submitted or issued.
void vk_mem_protect_until(uint64_t timeline_value);
void vk_mem_collect(void);                         // frees what has completed
typedef struct { uint64_t imported_pools, physical_pools, virtual_pools, import_failures, upload_batches; } VkMemStats;
void vk_mem_stats(VkMemStats *s);
void vk_mem_gpu_summary(char *out, size_t n);      // "; gpu: ..." for mem_commit_report: VMA, mirrors, staging, usnap
void vk_mem_shutdown(void);                        // before vk_shutdown: waits idle, frees every memory and texture object

// Textures. Images are keyed by storage GPU address plus descriptor and stay in VK_IMAGE_LAYOUT_GENERAL. A mirrored
// texture (CPU-written pool storage, MNG tex_mirrored) is uploaded on first use and again when its storage pages were
// written; once the GPU wrote it (vk_texture_gpu_written) the image is its content. A returned pointer stays valid
// until the next call that may drop textures (release, pool fini); copy what outlives that.
typedef struct {
    VkImage image;
    VkFormat format;               // after the fallbacks of tex_vk_format (ASTC -> RGBA8, RGB -> RGBA)
    VkImageType type;
    VkImageViewType view_type;     // view of the whole image
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
    uint64_t addr;                 // storage GPU address
    uint8_t alpha_one;             // X format: sample alpha as 1 through the view swizzle
    uint8_t chan[4];               // TexVkFormat.chan: component order of a stand-in format, applied by the view swizzle
    uint8_t mirrored, gpu_owned;
    uint8_t bound;                 // aliases physical-pool memory (physical and virtual pools)
    uint8_t block_views;           // BC image that renders through uncompressed views of its blocks (BLOCK_TEXEL_VIEW_
                                   // COMPATIBLE + EXTENDED_USAGE: only those views may use COLOR_ATTACHMENT)
    uint8_t swz[4];               // NVN texture swizzle selectors (nvnTextureBuilderSetSwizzle); a view swizzle replaces them
    uint8_t fb_usage;              // created with ATTACHMENT_FEEDBACK_LOOP usage (VK_FEEDBACK_LAYOUT, attachable images)
    uint8_t fb_learned;            // a draw sampled it while it was a written attachment of the open pass: its passes
                                   // attach it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT (vk_exec.c vk_exec_pass_begin).
                                   // Read and set through vk_tex_fb_learned and vk_tex_fb_learn
} VkTexImage;
// fb_learned is set by the replay thread (vk_exec_draw.c feedback_prepare) while the helper threads (vk_spec.h) read
// it for the draws they translate and the passes they begin, so it is loaded and stored atomically. A helper part that
// read it before the store is dropped: the store is followed by vk_spec_world_changed.
static inline int vk_tex_fb_learned(const VkTexImage *img) { return __atomic_load_n(&img->fb_learned, __ATOMIC_ACQUIRE); }
static inline void vk_tex_fb_learn(VkTexImage *img) { __atomic_store_n(&img->fb_learned, 1, __ATOMIC_RELEASE); }
const VkTexImage *vk_texture_get(const tex_t *t);   // created and made current; NULL for BUFFER or unmapped formats
// A texture slot; a slot whose texture was finalized follows the live texture at the same storage (hle_nvn.c).
const VkTexImage *vk_texture_by_id(const dpool_ref *pool, int id, view_t *view, int *has_view);
// The replay looks up many slots per draw: it takes the texture lock once around them (vk_texture_lock) and uses the
// _locked lookups; vk_texture_slots_gen changes whenever a slot may resolve to another image or view.
void vk_texture_lock(void);
void vk_texture_unlock(void);
const VkTexImage *vk_texture_by_id_locked(const dpool_ref *pool, int id, view_t *view, int *has_view);
VkSampler vk_sampler_by_id_locked(const dpool_ref *pool, int id);
// The slot's sampler with nearest filtering and no anisotropy: a depth texture read without a compare lands on one texel
// on the Switch although the game binds a linear sampler, and MacNativeGames samples it the same way. VK_NULL_HANDLE
// for an empty slot.
VkSampler vk_sampler_point_by_id_locked(const dpool_ref *pool, int id);
uint32_t vk_texture_slots_gen(void);
// What vk_texture_by_id(id) returns can change only when this does: the slot is registered again, or a texture entry at
// its storage address is created, dropped, released or taken back into use. Unlocked; read it before the lookup it
// guards (a lookup may create an entry). 0 when the slot was never registered: nothing is kept under 0.
uint64_t vk_texture_slot_token(const dpool_ref *pool, int id);
// What a token was read from, so that it can be checked again without the texture lock: vk_texture_slot_token_again
// gives the slot's token while it was not registered again or retired since (the address generation read anew), else 0.
typedef struct { uint32_t serial, ridx, rgen, aidx, valid; } VkSlotStamp;
uint64_t vk_texture_slot_token_stamped(const dpool_ref *pool, int id, VkSlotStamp *st);
uint64_t vk_texture_slot_token_again(const VkSlotStamp *st);
uint32_t vk_sampler_gen(void);                      // changes whenever a sampler slot is registered
void vk_texture_gpu_written(const tex_t *t);        // rendered or copied into by the GPU: no re-upload from storage
// The same for a part the helper thread recorded (vk_spec.h): _ready asks, on the helper, whether the marking can wait
// until the replay thread takes the part; _late does it then, and only for an entry that is still GPU-owned.
int vk_texture_gpu_written_ready(const tex_t *t);
void vk_texture_gpu_written_late(const tex_t *t);
void vk_texture_storage_bound_locked(const VkTexImage *img);   // bound as a storage image in the recording submission
uint64_t vk_texture_last_write(const tex_t *t);    // timeline value of the newest submission recorded to write it; 0: none
size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img);   // bytes ReadTexels spans; 0 for a bad region
int vk_texture_small_dirty(void);                   // a small mirrored target holds GPU texels its storage lacks
void vk_texture_write_back_small(uint64_t fence);   // writes those rendered by work up to fence back to storage
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written);   // vk_dump
int vk_texture_slot_swz(const dpool_ref *pool, int id, uint8_t swz[4]);
int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                          uint64_t img);            // always from the image (GPU copy, waited for); 0 on failure
int vk_texture_cpu_stale(uint64_t addr, uint64_t size);    // a GPU-written mirrored texture there has stale storage
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size);   // newest recorded writer of such a texture; 0: none
void vk_texture_write_back(uint64_t addr, uint64_t size);   // writes those images back into their storage (waits)
// The replay's deferred-read write-back (TOTK_WB_ASYNC, default off): copies recorded and submitted without waiting,
// written to the storage by a later vk_texture_wb_poll once they completed. TOTK_WB_ASYNC=0: vk_texture_write_back.
void vk_texture_write_back_async(uint64_t addr, uint64_t size);
void vk_texture_wb_poll(void);                      // finishes the asynchronous write-backs whose copies completed
void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc);   // vk_pool_fini: drop the images there
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads);
// Entries now, live and released (TextureFinalize, kept until sweep_released), with the bytes of their own memory
// (images not bound into pool memory).
typedef struct { uint64_t live, released, own_live, own_released, bytes_live, bytes_released; } VkTexCensus;
void vk_texture_census(VkTexCensus *s);
void vk_texture_shutdown(void);                     // vk_mem_shutdown calls it
void vk_texture_new_submit(void);                   // a queue submit starts: draw bindings check storage again
uint32_t vk_texture_epoch(void);                    // the write-watch snapshot epoch vk_texture_new_submit started
VkSampler vk_sampler_by_id(const dpool_ref *pool, int id);
VkSampler vk_sampler_for(const samp_par *s);        // cached by content
int vk_sampler_params(const dpool_ref *pool, int id, samp_par *out);
// DLSS in place of the game's upscaler (vk_dlss.c, upscaler = dlss, experimental, NVIDIA on Windows). vk_dlss_wanted:
// the setting asks for it and nothing has failed yet. vk_dlss_instance_exts / vk_dlss_device_exts: the extensions NVIDIA
// NGX asks for (at most max into out; the instance call loads the NGX core first). vk_dlss_record: DLSS of in (sampled,
// GENERAL) into out (storage, GENERAL), recorded into cb, the replay's command buffer, behind the caller's full barrier;
// fence is the value its submission signals. 1 = recorded (the caller forgets the bound state and puts a full barrier
// after it); 0 = nothing recorded, the game's upscaler runs (after a failure for the rest of the run); -1 = NGX failed
// after it may have recorded its own binds: the caller forgets the bound state and binds again before the game's one.
int vk_dlss_wanted(void);
int vk_dlss_instance_exts(const char **out, int max);
int vk_dlss_device_exts(const char **out, int max);
int vk_dlss_record(VkCommandBuffer cb, uint64_t fence, const VkTexImage *in, const VkTexImage *out);
void vk_dlss_release(void);   // device idle, before it goes away

// ---- submission, synchronisation, completion (vk_sync.c, vk_backend.c)
// Fence values are ctx->timeline values: each submission signals the next one, in submit order. A value counts as done
// once the GPU reached it and the completion thread ran its side effects. Guest or host threads.
void vk_sync_start(void);                  // vk_init (fatal without a device) and the completion thread; idempotent
uint64_t vk_sync_submit(const uint64_t *handles, int n, uint64_t submitted_ticks); // immutable front-end entry tick
uint64_t vk_sync_reserve(void);            // asynchronous replay: the value the next queued submission signals
int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns);   // 1 when done; 0 on timeout or a value never submitted
uint64_t vk_sync_fence_done(void);         // highest done value
uint64_t vk_sync_issued(void);             // newest submitted value (the one being recorded is this + 1)
void vk_sync_finish(void);                 // every submitted value done
void vk_sync_shutdown(void);               // finish, stop the completion thread, free the command buffers
// The op-stream replay (vk_exec.c): records the command handles of one submission into cb, which the caller has
// begun and ends. fence is the value this submission will signal. Returns the number of commands recorded into the
// command buffer it holds at the end (0: that one stays empty and is not submitted). A weak vk_sync.c version accepts
// empty streams only.
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence, uint64_t submitted_ticks);
// Record internal GPU work through the same submission and completion lifetime as NVN command handles. CPU
// recording is synchronous. Only the queue owner or an exclusively drained/idle backend may call it, without
// simultaneous producers. Preserve queued-fence order; never nest submission or replay inside the callback.
typedef int (*VkSyncRecordFn)(VkCommandBuffer cb, uint64_t fence, uint64_t submitted_ticks, void *context);
uint64_t vk_sync_submit_record(VkSyncRecordFn record, void *context, uint64_t submitted_ticks);
// Inside vk_exec_record, outside a render pass: ends the command buffer recorded so far, submits it (after the upload
// batch) without a signal, and returns a new begun one for the rest of the submission. The submission's timeline value
// is signalled by its last part, whose signal covers every earlier part in submission order.
VkCommandBuffer vk_sync_split(void);
// Inside vk_exec_record, right after vk_sync_split: submits cb, a finished command buffer the helper thread recorded
// (vk_spec.h), as the next part of the submission, after the upload batch and the mirror copies it needs. No signal.
void vk_sync_split_insert(VkCommandBuffer cb);
// Submit the recorded prefix and wait for its dedicated fence, without advancing the full submission's timeline.
// Replay thread, outside a render pass. Returns a fresh recording command buffer; old resources keep their lifetime.
VkCommandBuffer vk_sync_split_wait(void);
// TOTK_TRACE=perf: the replay notes each draw or dispatch of the part being timed (kind 0 draw, 1 indirect draw, 2
// dispatch, 3 indirect dispatch; a, b: the NVN 2 names of its vertex and fragment stages, or its compute stage and 0;
// work: vertices x instances or groups, 0 when indirect), so a slow part can name its work. vk_sync_noting: 1 while
// the part is timed. vk_sync_note_args: the indirect draw just noted takes its VkDrawIndexedIndirectCommand from this
// host address (an imported pool); a slow part reads them back once the GPU is done with it. vk_sync_note_write: a
// stage named `stage` stores into [addr, addr+size) of guest memory, so a slow part can tell which shader of the part
// wrote its largest indirect draw's arguments.
int vk_sync_noting(void);
void vk_sync_note(int kind, uint64_t a, uint64_t b, uint64_t work);
void vk_sync_note_args(uint64_t addr);
void vk_sync_note_write(uint64_t addr, uint64_t size, uint64_t stage);
// Runs fn(arg) on the completion thread once fence is done on the GPU (event writes, counter reports, sync state).
// Callbacks run in the order they were added, after those of earlier values; a value already done runs at once there.
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg);
// gfx_vulkan_backend (vk_backend.c): presents go to a present thread and run after the submissions made before them.
void rdoc_init(void);                     // runtime/rdoc.c: TOTK_RDOC debug capture; before vkCreateInstance
int rdoc_active(void);
void rdoc_after_present(uint64_t frame);
void vk_present_drain(void);               // returns once every queued present was submitted to gfx_queue
int vk_replay_async(void);                 // TOTK_REPLAY_ASYNC (default 0): submissions replay on the replay thread
void vk_program_release(const prog_t *p);  // GfxBackend.program_release: vk_pipeline.c; weak no-op in vk_backend.c
void vk_backend_shutdown(void);            // presents, vk_sync_shutdown, vk_spec_shutdown, vk_mem_shutdown, vk_shutdown in that order

// ---- pipelines (vk_pipeline.c)
// Binding model (comment block at the top of vk_pipeline.c): one descriptor set, set 0, per pipeline. Binding 0 is
// Ryujinx's support buffer (a uniform buffer every stage shares); the vertex stage keeps its own bindings, the fragment
// stage's follow them (its SPIR-V is renumbered). Set 0 is a push-descriptor set unless the pipeline needs more than
// maxPushDescriptors (32 here) descriptors; VkPipeInfo.bindings says what goes where. No push constants.
// Everything a graphics pipeline depends on, as the op stream carries it. Fields covered by dynamic state are read by
// vk_pipeline_set_dynamic only and left out of the pipeline key.
typedef struct VkPipeState {
    const GfxShader *vs, *fs;              // gfx_shader_get results; fs NULL: no fragment stage (depth only)
    int32_t color_fmt[8];                  // NVNformat of colour target i (OP_SET_RT), 0 none
    int32_t depth_fmt;                     // NVNformat of the depth/stencil target, 0 none
    uint8_t samples;                       // sample count of the targets; 0 or 1 single-sampled
    uint8_t prim;                          // NVN_DRAW_PRIMITIVE_* of the draw
    uint8_t depth_minus_w;                 // nvn_depth_minus_w(): clip-space depth is -w..w
    uint8_t depth_clamp, raster_discard;   // OP_DEPTH_CLAMP, OP_RASTER_DISCARD
    uint8_t y_flip;                        // the replay's viewports have negative heights (upper-left origin): the
                                           // front face is not mirrored
    uint8_t prim_restart;                  // the draw's indices restart its strip or fan at the all-ones value
                                           // (OP_PRIM_RESTART, set by op_draw_idx); dynamic state only
    uint8_t feedback;                      // the fragment shader samples an attachment of the pass in the feedback
                                           // layout: 1 colour, 2 depth/stencil (VK_PIPELINE_CREATE_*_FEEDBACK_LOOP_BIT_EXT)
    blend_st blend[8];                    // OP_BIND_STATE kind 0, indexed by blend_st.target
    color_st color;                        // kinds 1-5
    chmask_st chmask;
    ms_st ms;
    poly_st poly;
    ds_st ds;
    uint32_t vattr[16];                    // OP_BIND_VATTR: va_st words; attribute i feeds vertex shader Location i
    vs_st vstream[16];                     // OP_BIND_VSTREAM
    // Keep the legacy fields above byte-for-byte stable for pipeline-key records.
    const GfxShader *gs;                   // experimental ordinary GS; NULL keeps the existing VS/FS paths
    float alpha_ref;                       // OP_BIND_STATE kind 7: reference of color.alpha_func's test. After gs, so not in
                                           // the pipeline-key records: a prebuilt alpha-test pipeline has reference 0 and
                                           // goes unused, and the draw builds its own
} VkPipeState;
void vk_pipe_state_defaults(VkPipeState *s);   // NVN defaults of every state object (shaders and formats left 0)
enum { VK_PIPE_SUPPORT = GFX_RES_KINDS };      // VkPipeBinding.kind of the support buffer
#define VK_PIPE_ZERO_BINDING 16                // vertex binding of shader inputs without an attribute: bind a zeroed
                                               // buffer of at least 16 bytes there (stride 0)
typedef struct {
    uint16_t binding;                      // in its set
    uint8_t stage;                         // GFX_STAGE_* that declared it (support buffer: the first stage)
    uint8_t kind;                          // GFX_RES_* or VK_PIPE_SUPPORT
    VkDescriptorType type;
    uint16_t count;
    VkShaderStageFlags stages;
    const GfxShaderRes *res;               // the stage's reflection entry; NULL for the support buffer
    uint8_t set;                           // descriptor set: 0, or 1 for the fragment stage with independent layouts
} VkPipeBinding;
typedef struct {
    VkPipeline pipeline;                   // VK_NULL_HANDLE when creation failed
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout;      // set 0
    int push;                              // 1: write set 0 with vkCmdPushDescriptorSetKHR; 0: allocate and bind a set
    VkPipelineBindPoint bind_point;
    uint32_t n_bindings;
    const VkPipeBinding *bindings;
    uint32_t vs_inputs;                    // Locations the vertex shader reads
    // Independent layouts (vk_caps.gpl): set 0 holds the vertex stage's bindings, set 1 the fragment stage's; bindings
    // are ordered by set. One of the two sets at most is a push set. n_sets 1: everything in set 0 (TOTK_GPL=0, compute).
    VkDescriptorSetLayout set_layout1;
    int push1;
    uint32_t n_sets;
    // Descriptor update template per set (vk_pipeline.c make_tpl): reads one VkDescSlot per descriptor, in binding order,
    // from the start of the set's descriptors. VK_NULL_HANDLE: the set is written with VkWriteDescriptorSet structs.
    VkDescriptorUpdateTemplate tpl[2];
} VkPipeInfo;
// One descriptor of the packed array a descriptor update template reads (VkPipeInfo.tpl), 24 bytes for every type; the
// bytes a type leaves unused are zero, so equal descriptors compare equal byte for byte.
typedef union { VkDescriptorImageInfo image; VkDescriptorBufferInfo buffer; VkBufferView view; uint64_t w[3]; } VkDescSlot;
// The most uniform buffers of one set the pipelines make UNIFORM_BUFFER_DYNAMIC (TOTK_DYN_UBO, vk_pipeline.c).
#define VK_DYN_UBO_MAX 32
// Both return the cached pipeline (created on first use; failures are cached and logged once). info: may be NULL.
// Valid until vk_pipeline_shutdown; thread-safe.
VkPipeline vk_pipeline_graphics(const VkPipeState *s, const VkPipeInfo **info);
// The replay's variant: after vk_pipeline_async_enable a missing pipeline is built by background workers and this
// returns VK_NULL_HANDLE with *pending 1 until it is ready (the draw is skipped); otherwise as vk_pipeline_graphics.
// wait_ms > 0 waits up to that long for a pending pipeline first. Pipelines built in a run are recorded to
// pipeline-keys.bin next to the pipeline cache and prebuilt at the next boot (TOTK_PIPE_PREBUILD=0: not).
VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms);
// rt_boot calls it unless --shot is given; TOTK_PIPE_SYNC=1 makes it a no-op. Probes never call it.
void vk_pipeline_async_enable(void);
int vk_pipeline_async_on(void);   // 1 after vk_pipeline_async_enable: graphics shaders are translated in the background too
// Boot prewarm, before the guest runs (rt_boot): loads the shaders pipeline-keys.bin names, translating the ones not
// cached under the current translator from their stored inputs, and builds every recorded pipeline, on all cores.
// No-op without vk_pipeline_async_enable, with TOTK_PIPE_PREBUILD=0 or with TOTK_GPU=null.
void vk_pipeline_prewarm(void);
VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info);
// Records every dynamic state that VkPipeState determines, after binding a graphics pipeline: cull mode, front face,
// topology, depth and stencil tests and ops, depth bias enable, rasterizer discard, vertex input, and, where the GPU
// has them dynamic, depth clamp, polygon mode, blend enables, equations, write masks and alpha to coverage. The replay
// sets the rest: vkCmdSetViewportWithCount, vkCmdSetScissorWithCount, depth bias, blend constants, stencil compare and
// write masks and references; vertex buffers with vkCmdBindVertexBuffers2 (strides NULL when
// vk_pipeline_vertex_input_dynamic(), else from VkPipeState.vstream).
void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s);
int vk_pipeline_vertex_input_dynamic(void);    // 1: vertex input is VK_EXT_vertex_input_dynamic_state
// Translator answers from the bound state: attr_type from the vertex attribute formats (vs), frag_out_type from the
// colour target formats (fs). Other fields are left as they are; either pointer may be NULL.
void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs);
int vk_pipeline_tex_type(int nvn_target);      // NVN_TEXTURE_TARGET_* -> GFX_TEX_* (GfxShaderAnswers.tex_type)
uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i);   // colour write mask of target i the pipeline uses
VkPrimitiveTopology vk_pipeline_topology(int nvn_prim);   // quads -> triangle list, line loop -> strip: the replay
                                                          // rewrites those draws; polygon -> fan
int vk_geometry_supported(void);                   // 1: the device has geometryShader (MoltenVK does not)
uint8_t vk_geometry_input(uint32_t nvn_primitive);    // GFX_GEOMETRY_*; 0 for unsupported/emulated input primitives
// TOTK_DUMP=key (vk_dump.c): F9 in the game window dumps the frame after next; the window calls this on the key.
void vk_dump_request(void);
VkFormat vk_vertex_format(int nvn_format, int *type);     // attribute format (supported fallback), GFX_TYPE_* in *type
PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void);  // after the first vk_pipeline_* call
// Descriptor update templates (after the first vk_pipeline_* call): NULL when templates are off (TOTK_DESC_TEMPLATE=0,
// a device below Vulkan 1.1) or, for the push variant, vkCmdPushDescriptorSetWithTemplateKHR is missing.
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void);
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void);
// Probe only: 1 selects stage-independent layouts for pipelines looked up from now on, 0 the shared set 0; returns
// the previous mode. 1 needs vk_caps.gpl (the independent-sets flag comes with the extension).
int vk_pipeline_set_indep(int on);
typedef struct { uint64_t graphics, compute, failures, hits, layouts, push_layouts, async_built, pending_skips; } VkPipeStats;
void vk_pipeline_stats(VkPipeStats *s);
void vk_pipeline_save(void);               // writes the VkPipelineCache to pipeline-cache.bin next to totk.exe
void vk_pipeline_autosave(void);           // vk_pipeline_save at most every 20 s, when pipelines were built since
void vk_pipeline_shutdown(void);           // waits for background builds, saves, destroys pipelines, layouts and the
                                           // cache; before vk_shutdown
int vk_pipeline_selftest(void);            // one graphics and one compute pipeline from census shaders; 0 on success

// Original fixed-function clear helper. Caller supplies a one-layer view and orders it with surrounding work.
void vk_clear_stencil_masked(VkCommandBuffer cb, VkFormat format, uint32_t samples, VkImageAspectFlags aspect,
                             VkImageView view, uint32_t width, uint32_t height, VkRect2D scissor,
                             uint32_t mask, uint32_t value);
void vk_clear_shutdown(VkDevice device);   // after device idle, before destroying the device

#include "vk_rec.h"   // last: its macros route the vkCmd* calls of every file that includes vk.h
