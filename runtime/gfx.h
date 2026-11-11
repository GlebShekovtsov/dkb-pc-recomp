// Interface of the graphics layer: window, shader service, back ends and the NVN front end.
#pragma once
#include <stdint.h>
#include "host.h"
#include "nvn_obj.h"
#include "nvn2_c0.h"

// ---- moved from ov.h
enum { OE_MSG_FOCUS_STATE_CHANGED = 15, OE_MSG_OPERATION_MODE_CHANGED = 30, OE_MSG_PERFORMANCE_MODE_CHANGED = 31 };
void oe_push_message(uint32_t message);                   // ov_system.c
void oe_set_focus(int in_focus);                          // ov_system.c: GetCurrentFocusState; TOTK_ASSUME_FOCUS pins 1
typedef struct { uint64_t alloc, free, realloc, user; } OvNvAllocator;
typedef struct { OvNvAllocator graphics, devtools; uint64_t memory, memory_size; int initialized; } OvNvState;
const OvNvState *ov_nv_state(void);                        // ov_gfx.c

// ---- GPU address window for CPU-invisible pools (section 3.1)
#define RT_GPU_VA_BASE 0x8000000000ull                     // 2^39, never mapped on the host
#define RT_GPU_VA_SIZE (64ull << 30)

// ---- window, vsync, focus (gfx_window.c, the window back end gfx_wsys.h and the chosen back end). Host threads only;
// never run guest code.
// Before plat_exit, after the saves: the back end's queued present goes out (exit_wait, outside the start lock: the
// thread presenting it may be waiting for that lock), then the device's hold on the window is released (exit_release,
// under it); parks if another exit is under way.
void gfx_exit_release(void);
void gfx_window_start(void);                               // the back end's device, window and swapchain; idempotent
void gfx_window_resize(int client_w, int client_h);        // programmatic resize; returns once the back end followed it
int gfx_window_exiting(void);                              // an exit released the device: present and start drop out
int gfx_window_closing(void);                              // the exit has started: the swapchain and the present passes are being freed
int gfx_window_dirty(void);                                // the window changed size since the last take; leaves the mark
int gfx_window_take_dirty(void);                           // the window changed size since the last call; clears the mark
// The rectangle {x, y, w, h} a cw x ch picture is presented in inside a dw x dh image (TOTK_ASPECT); the rest is black.
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]);
#ifdef __APPLE__
// macOS runs SDL's video calls (and AppKit's) on the process's main thread, so that thread runs the window's event
// loop and the game boots on a thread of its own (boot.c rt_boot). Called on the main thread; does not return.
void gfx_window_main_thread(void);
#endif
void gfx_vsync_bind(uint64_t system_event);                // nn::vi vsync event; signalled at 60 Hz
uint64_t gfx_vsync_ticks(void);                            // ticks since start
uint64_t gfx_vsync_time_ns(uint64_t tick);                 // kern_now_ns() time of a tick (pacing)
double gfx_window_refresh_hz(void);                        // refresh rate of the window's monitor; 0 unknown
int gfx_window_hdr10(void);                                // the swapchain is HDR10 now; 0 before one and on SDR
int gfx_window_focused(void);                              // the window has the focus (nn::oe's focus state follows it)
unsigned gfx_window_focus_messages(void);                  // OE_MSG_FOCUS_STATE_CHANGED pushes made by the window thread

// ---- shader service client (gfx_shader.c)
enum { GFX_STAGE_VS, GFX_STAGE_TCS, GFX_STAGE_TES, GFX_STAGE_GS, GFX_STAGE_FS, GFX_STAGE_CS };
enum { GFX_TEX_1D = 1, GFX_TEX_1D_ARRAY, GFX_TEX_BUFFER, GFX_TEX_2D, GFX_TEX_2D_ARRAY, GFX_TEX_2D_MS,
       GFX_TEX_2D_MS_ARRAY, GFX_TEX_3D, GFX_TEX_CUBE, GFX_TEX_CUBE_ARRAY };   // gfx_shader.c tex_names order
enum { GFX_TYPE_FLOAT, GFX_TYPE_SINT, GFX_TYPE_UINT };
enum { GFX_GEOMETRY_NONE, GFX_GEOMETRY_POINTS, GFX_GEOMETRY_LINES, GFX_GEOMETRY_LINES_ADJACENCY,
       GFX_GEOMETRY_TRIANGLES, GFX_GEOMETRY_TRIANGLES_ADJACENCY };
enum { GFX_GEOMETRY_OUTPUT_POINTS = 1, GFX_GEOMETRY_OUTPUT_LINE_STRIP, GFX_GEOMETRY_OUTPUT_TRIANGLE_STRIP };
#define GFX_GEOMETRY_VERSION 1
#define GFX_GEOMETRY_PASSTHROUGH_VERSION 2
typedef struct {                                           // the translator's answers (Program.cs Accessor)
    uint8_t tex_type[32], img_type[32], img_format[32];    // per c2 slot; 0 = not asked; types are GFX_TEX_*,
                                                           // img_format is Ryujinx TextureFormat
    uint8_t attr_type[32], frag_out_type[8];               // GFX_TYPE_*
    uint16_t local[3]; uint32_t shared_bytes;
    uint8_t depth_minus_w;
    uint8_t geometry_input;                              // GFX_GEOMETRY_*; required for GS, zero for other stages
} GfxShaderAnswers;
typedef struct GfxShader GfxShader;                        // SPIR-V words + parsed reflection; owned by the cache
void gfx_shader_nvn2_dump(const uint8_t *control, const uint8_t *code);   // DKB_SHADER_DUMP (gfx_shader.c)
int gfx_shader_is_nvn2(const uint8_t *control);                          // NVN 2 control block (gfx_shader.c)
uint64_t gfx_shader_nvn2_hash(const uint8_t *control, const uint8_t *code);   // DKB_SHADER_DUMP name, 0 if not NVN 2
uint64_t nvn2_origin(const uint8_t *code);   // that name of the NVN 2 stage a lowered code buffer came from, or 0
// NVN 2 stage -> NVN 1 control and Maxwell code (nvn2_lower.cpp); 0 when it does not lower
int nvn2_lower(const uint8_t *control, const uint8_t *code, const uint8_t **out_control, const uint8_t **out_code);
const GfxShader *gfx_shader_get(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                const uint8_t *c1, const GfxShaderAnswers *a);   // blocks; NULL on failure (counted)
// As gfx_shader_get with c1 inside the code, but a shader in neither cache is queued to a background translation and
// NULL returned with *pending = 1 until it is ready. A failed translation stays failed (*pending = 0).
const GfxShader *gfx_shader_get_async(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                      const GfxShaderAnswers *a, int *pending);
// As gfx_shader_get_async, but a shader queued for translation is moved to the front of the queue and waited for.
// NULL when the translation failed.
const GfxShader *gfx_shader_await(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                  const GfxShaderAnswers *a);
uint64_t gfx_shader_async_done(void);                      // translations finished by the background worker
void gfx_shader_counts(uint64_t *keys, uint64_t *translated);   // distinct keys in memory, translations done (TOTK_TRACE=perf)
// Translator answers for this device (vk_caps.c): shaderFloat64, storage offset rounding, subgroup size per vs/fs/cs.
void gfx_shader_set_host(int float64, uint32_t sbuf_align, const uint32_t subgroup[3], uint32_t gather_bits);
// Both host setters precede shader requests. GS has no subgroup default; repeated identical setup is harmless.
void gfx_shader_set_geometry_host(uint32_t subgroup);
const uint8_t *gfx_shader_key(const GfxShader *s);          // the 32-byte cache key (stable while the translator is)
const uint8_t *gfx_shader_input_id(const GfxShader *s);     // the 32-byte input id (stable across translators)
// By input id from memory or disk; xlate 1 also translates it from its stored inputs. NULL if absent.
const GfxShader *gfx_shader_load_input(const uint8_t iid[32], int xlate);
// Boot prewarm: out[i] = gfx_shader_load_input(iids[i], 1) on `threads` threads with up to `procs` translators, then
// every other stored input not cached under this translator is translated. Logs progress and the total.
void gfx_shader_prewarm(const uint8_t (*iids)[32], size_t n, const GfxShader **out, int threads, int procs);
// Answers met in play: from this call on, every (program, answers) a gfx_shader_get* call resolves is appended
// to shader-answers.bin, and its program to shader-programs.bin, in the directory of `keys_path` (pipeline-keys.bin).
void gfx_shader_record_at(const char *keys_path);
void gfx_shader_record_flush(int force);                   // appends pending records; without force at most every 2 s
// Translates every recorded (program, answers) on `threads` threads with up to `procs` translators (TOTK_TEST_PREPARE_SHADERS).
void gfx_shader_prepare_recorded(int threads, int procs);
// totk.exe --prepare-shaders: translates <dir>/seed-answers.bin + seed-programs.bin (tools/prepare_shaders.py) and
// writes <dir>/shader-prepared.bin (stage, priority, input id per variant). 0 on success.
int gfx_shader_prepare_seed(const char *dir, int threads, int procs);
typedef struct { uint8_t iid[32]; uint8_t stage, prio; } GfxWarmEnt;   // prio 0: met in play; 1..3: prepared
size_t gfx_shader_warm_list(GfxWarmEnt **out);              // malloc'd; recorded first, then prepared; needs recording on
// Disk-cache load the caller owns (*owned 1: release it), or the memory cache's entry (*owned 0). NULL if not cached.
const GfxShader *gfx_shader_load_private(const uint8_t iid[32], int *owned);
void gfx_shader_release_private(const GfxShader *s);
const uint32_t *gfx_shader_spirv(const GfxShader *s, uint64_t *words);
uint64_t gfx_shader_spirv_fnv(const GfxShader *s);          // FNV-1a 64 of the SPIR-V bytes, taken once per shader
uint64_t gfx_shader_failures(void);                         // failed translations; feeds GfxStats.shader_failures
// Reflection: parsed once per shader from the translator's .res.json and the SPIR-V before it enters the cache.
enum { GFX_RES_UBO, GFX_RES_SSBO, GFX_RES_TEX, GFX_RES_IMG, GFX_RES_KINDS };
// Version 1 requires guarded SSBOs to expose zero array elements when their address cannot be bound.
enum { GFX_STORAGE_CANDIDATES_VERSION = 1, GFX_RES_SSBO_GUARDED = 0x80 };
typedef struct {
    uint16_t set, binding;      // as the SPIR-V decorates them: set 0, bindings from 1 per shader (0: support buffer)
    int16_t slot;               // UBO: c[slot]; SSBO: storage buffer index; TEX: c2 combined slot (handle at 0x20 + 8 slot)
                                // or, separate, texture slot (0x168 + 8 slot); IMG: image slot (0x120 + 8 slot); -1: other
    uint16_t type;              // TEX/IMG: Ryujinx SamplerType: 1 1D, 2 buffer, 3 2D, 4 3D, 5 cube | 0x100 array,
                                // 0x200 multisample, 0x400 shadow
    uint8_t format;             // IMG: declared format as a GfxShaderAnswers.img_format code, 0 unknown
    uint8_t flags;              // SSBO: 1 written, GFX_RES_SSBO_GUARDED; TEX/IMG: Ryujinx TextureUsageFlags
    int8_t cbuf_slot;           // TEX/IMG: constant buffer of the handle (-1: bound texture buffer); SSBO: of its address
    uint16_t handle;            // TEX/IMG: raw handle (word offset; separate: texture | sampler << 14); SSBO: address offset
    uint16_t count;             // descriptors (array length)
    uint8_t separate;           // TEX: texture and sampler handles are separate
} GfxShaderRes;
typedef struct { uint8_t type, comps, packed, scalar_bits; } GfxShaderIo;   // scalar_bits: actual numeric SPIR-V width, 0 unknown
uint64_t gfx_shader_origin(const GfxShader *s);             // NVN 2 stage origin: noted when its code was lowered, in this run or
                                                            // an earlier one (shader-cache/nvn2-origins.bin), or 0
int gfx_shader_stage(const GfxShader *s);                  // GFX_STAGE_VS, GFX_STAGE_GS, GFX_STAGE_FS or GFX_STAGE_CS
typedef struct {
    uint32_t version, input, input_vertices, output, invocations, max_output_vertices, passthrough;
} GfxShaderGeometry;
// Ordinary GS contract; NULL for another stage. Translation support does not imply replay/pipeline support.
const GfxShaderGeometry *gfx_shader_geometry(const GfxShader *s);
int gfx_shader_resources(const GfxShader *s, int kind, const GfxShaderRes **res);   // count of GFX_RES_* kind; array in *res
// NVN 2 compute parameter-block layout, including the declared root-table word counts, or NULL.
const Nvn2C0Layout *gfx_shader_c0_layout(const GfxShader *s);
uint32_t gfx_shader_slot_mask(const GfxShader *s);                    // texture/image slots whose type its translation read
uint32_t gfx_shader_inputs(const GfxShader *s, GfxShaderIo io[32]);    // Input Locations (VS attributes, FS varyings) as a mask
uint32_t gfx_shader_outputs(const GfxShader *s, GfxShaderIo io[32]);   // Output Locations (VS varyings, FS colour targets)
void gfx_shader_local_size(const GfxShader *s, uint32_t local[3]);     // CS: the LocalSize execution mode; 1, 1, 1 otherwise

// ---- back end (gfx_null.c, vk_*.c).
// TOTK_TRACE=gfx and the flight recorder: replay counters of the current frame, as the back end counts them.
typedef struct {
    uint64_t passes, draws_seen, draws, clears, copies, dispatches, submits;
    uint64_t fb_draws, fb_barriers, fb_restarts, fb_layouts;   // feedback loops: draws sampling a written target of
                                                               // their pass, in-pass barriers, pass restarts, layout
                                                               // transitions
} GfxExecFrame;
enum { GFX_TOOL_PREPARE_SHADERS, GFX_TOOL_TRANSLATE_SHADERS, GFX_TOOL_WARM_SHADERS };   // GfxBackend.shader_tool
typedef struct {
    const char *name;
    // ---- guest side: called by the NVN front end (nvn_api_*.c, nvn_ext.c) on guest threads.
    int  (*pool_init)(pool_t *p);                          // 0 fails MemoryPoolInitialize
    void (*pool_fini)(pool_t *p);
    int  (*pool_map_virtual)(pool_t *v, int n, const void *requests);   // NVNmappingRequest[n], 0x28 bytes each
    void (*texture_register)(const dpool_ref *pool, int id, const tex_t *t, const view_t *v);
    void (*sampler_register)(const dpool_ref *pool, int id, const samp_par *s);
    void (*texture_release)(const tex_t *t);
    void (*program_release)(const prog_t *p);
    // submitted_ticks is the front end's immutable entry tick, including 0; it is not GPU execution time.
    uint64_t (*submit)(const uint64_t *handles, int n, uint64_t submitted_ticks); // returns this submission's fence
    int  (*fence_wait)(uint64_t value, uint64_t timeout_ns);   // 1 reached; caller has given its core up
    uint64_t (*fence_done)(void);                          // highest value whose side effects are written
    // Queues the present and returns at once; the host present goes out at at_ns (kern_now_ns clock), at once when that
    // has passed or at_ns is 0.
    void (*present)(const tex_t *t, const int crop[4], uint64_t at_ns);
    void (*finish)(void);
    void (*read_texels)(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
    void (*write_texels)(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img);
    void (*guest_written)(uint64_t addr, uint64_t size);   // host wrote pool storage (3.2)
    void (*guest_read)(uint64_t addr, uint64_t size);      // the CPU reads pool storage: GPU-written texels go there first
    // The CPU waited for fence: small render targets written by work up to it go to their storage (vk_texture.c
    // vk_texture_write_back_small).
    void (*cpu_synced)(uint64_t fence);
    // ---- host side: boot, the exit paths, the command-line tools, the window and diagnostics. Host threads, never
    // guest code. A back end without a device (null) does nothing in each.
    // gfx_window_start: the device, the window, its surface and the swapchain or drawable; idempotent.
    void (*window_start)(void);
    void (*window_resize)(void);    // gfx_window_resize: the window has its new size; the swapchain or drawable follows
    const char *(*present_mode)(void);   // gfx_window_present_mode: the mode the swapchain got; "" before one, "none" without
    int  (*window_hdr10)(void);     // gfx_window_hdr10: the swapchain is HDR10 now; any thread, guest threads included
    void (*boot_async)(void);       // boot, before the prewarm: pipelines may be built in the background (not with --shot)
    void (*boot_prewarm)(void);     // boot: what earlier runs recorded is made ready; the guest starts after it returns
    void (*save)(void);             // every exit path, with the saves: the back end's caches and records go to disk
    // gfx_exit_release, before the process ends. exit_wait runs before gfx_window.c's start lock: the present queued
    // last goes out, and the thread that presents it may be waiting for that lock (gfx_window_start). exit_release
    // runs under the lock: the device lets go of the window (swapchain, surface, present objects).
    void (*exit_wait)(void);
    void (*exit_release)(void);
    int  (*shader_tool)(int tool);  // --prepare-shaders and the like: GFX_TOOL_*; returns the process's exit code
    // TOTK_FLIGHT thumbnails (flight.h): each present also keeps a FLIGHT_THUMB_W x FLIGHT_THUMB_H copy in a ring
    // of n. thumbs_enable before the first present. thumbs_read copies frame's as RGBA when the ring still holds it
    // complete; 0 when it was never written, is still in flight or was overwritten. Any thread.
    void (*thumbs_enable)(int n);
    int  (*thumbs_read)(uint64_t frame, uint8_t *rgba);
    const GfxExecFrame *(*exec_frame)(void);   // the replay counters; they grow until the back end's frame log resets them
    int  (*device_text)(char *buf, size_t n);  // the system info's GPU and driver lines; 0 while there is no device
    void (*dump_request)(void);     // F9 in the game window: the frame dump of TOTK_DUMP=key
} GfxBackend;
extern const GfxBackend gfx_null_backend, gfx_vulkan_backend;
const GfxBackend *gfx_backend(void);                       // TOTK_GPU=null|vulkan (default vulkan)
int gfx_shader_tool(int tool);                             // the chosen back end's shader_tool; main.c's --*-shaders modes

// ---- NVN front end (nvn_procs.c)
void nvn_procs_register(void);                             // hostfn_register every proc; before hostfn_seal (ov_gfx.c)
uint64_t nvn_proc_address(const char *name);               // synthetic host address of a proc, 0 for an unknown name

// ---- statistics and frames (nvn_api_*.c, vk_*.c; boot.c prints them)
typedef struct { uint64_t frames, nvn_calls, nvn_distinct, unimplemented, validation_errors, shader_failures,
                 draws_skipped; } GfxStats;
GfxStats gfx_stats(void);
void rt_frame_presented(const tex_t *t, uint64_t n);       // boot.c: --frames N ends the run after the Nth present
                                                           // (saves --shot first through read_texels)
