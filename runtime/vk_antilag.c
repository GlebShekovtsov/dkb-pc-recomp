// AMD Anti-Lag 2 through VK_AMD_anti_lag, without AMD's SDK (TOTK_ANTILAG = 1; off by default, experimental): the
// driver delays the start of the game's next frame so that its input is sampled later and the frame waits less in the
// queue, for less delay from input to the screen. A C rewrite of Hookshot's gpu/vk/vk_antilag.cpp
// (https://git.notrexed.dev/Hookshot/Hookshot @ 7191250b7b0f8a4becd79ee086229a2620872fc7, GPL-3.0).
//
// While the setting is on, each frame tells the driver two points of its life with vkAntiLagUpdateAMD, both with the
// frame's index: its input stage after the game presented the frame before (vk_backend.c vk_be_present, where the
// driver waits until the frame should start), and its present stage right before the present of the frame on the
// present thread (vk_backend.c present_main, which presents with vk_present_flip). Off is the game as without it: the
// driver is told Off once, nothing else is called. The extension is enabled whenever the device has it (vk_device.c),
// so the setting needs no restart: enabled alone it does nothing.
//
// The frames are shown by the present thread, later than the game's next input stage, so each present job keeps the
// game frame it was queued at (vk_antilag_frame) and gives that index at its present. With frame generation only the
// game's own frame gives a present stage (vk_framegen.c vk_fg_show); the made frame gives none.
//
// The input stage stands for the game's simulation start only roughly: it is given on the thread that presents, and
// under asynchronous replay (TOTK_REPLAY_ASYNC) the game has by then run up to a frame ahead.
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "platform/platform.h"

#ifdef VK_AMD_anti_lag

static struct {
    int tried, ready;
    PFN_vkAntiLagUpdateAMD update;
    int active;         // the driver was last told On
    int frame_marked;   // this frame's input stage was given: its present stage counts
    uint64_t frame;     // the frame's index, the same for both its stages
    uint64_t since;     // the last index before the setting last changed: frames up to it get no present stage
} A;
static PlatMutex al = PLAT_MUTEX_INIT;

// With al held: the function, once the device has the feature.
static int ready(void) {
    if (A.tried) return A.ready;
    VkCtx *c = vk_ctx();
    if (!c || !c->device) return 0;   // asked before the device: not decided yet
    A.tried = 1;
    if (!c->has_anti_lag) return 0;
    A.update = (PFN_vkAntiLagUpdateAMD)vkGetDeviceProcAddr(c->device, "vkAntiLagUpdateAMD");
    A.ready = A.update != NULL;
    rt_log("vk: AMD Anti-Lag 2: %s", A.ready ? "available (VK_AMD_anti_lag)" : "its function is missing: off");
    return A.ready;
}

// The mode to the driver, with one of the frame's stages or none (the mode alone), through fn on dev; nothing without
// them (no device, or the module reset by vk_antilag_release).
static void update_on(PFN_vkAntiLagUpdateAMD fn, VkDevice dev, int on, const VkAntiLagPresentationInfoAMD *stage) {
    if (!fn || !dev) return;
    VkAntiLagDataAMD d = {.sType = VK_STRUCTURE_TYPE_ANTI_LAG_DATA_AMD,
                          .mode = on ? VK_ANTI_LAG_MODE_ON_AMD : VK_ANTI_LAG_MODE_OFF_AMD,
                          .maxFPS = 0,   // no limit of its own: the game's frame rate limit paces
                          .pPresentationInfo = stage};
    fn(dev, &d);
}

// With al held: the device of now, or VK_NULL_HANDLE between a vk_shutdown and the next device.
static VkDevice device_now(void) {
    VkCtx *c = vk_ctx();
    return c ? c->device : VK_NULL_HANDLE;
}

// With al held.
static void update(int on, const VkAntiLagPresentationInfoAMD *stage) { update_on(A.update, device_now(), on, stage); }

// With al held: the setting on, and the extension and its feature. Turning off tells the driver once.
static int on(void) {
    if (!ready()) return 0;
    int want = cfg_bool("TOTK_ANTILAG");
    if (want != A.active) {
        if (!want) update(0, NULL);
        A.active = want, A.frame_marked = 0, A.since = A.frame;
        rt_log("vk: AMD Anti-Lag 2: %s", want ? "on" : "off");
    }
    return want;
}

static void stage(VkAntiLagStageAMD s, uint64_t frame) {
    VkAntiLagPresentationInfoAMD p = {.sType = VK_STRUCTURE_TYPE_ANTI_LAG_PRESENTATION_INFO_AMD, .stage = s,
                                      .frameIndex = frame};
    update(1, &p);
}

int vk_antilag_available(void) {
    plat_mutex_lock(&al);
    int r = ready();
    plat_mutex_unlock(&al);
    return r;
}

uint64_t vk_antilag_frame(void) {
    plat_mutex_lock(&al);
    uint64_t f = on() && A.frame_marked ? A.frame : 0;
    plat_mutex_unlock(&al);
    return f;
}

void vk_antilag_present(uint64_t frame) {
    if (!frame) return;   // a made frame, or a frame queued without an input stage
    plat_mutex_lock(&al);
    // a frame from before the setting was last turned on had its input stage in the driver's earlier run
    if (on() && frame > A.since) stage(VK_ANTI_LAG_STAGE_PRESENT_AMD, frame);
    plat_mutex_unlock(&al);
}

void vk_antilag_input(void) {
    uint64_t frame = 0;
    PFN_vkAntiLagUpdateAMD fn = NULL;
    VkDevice dev = VK_NULL_HANDLE;
    plat_mutex_lock(&al);
    if (on()) {
        frame = ++A.frame;
        A.frame_marked = 1;
        fn = A.update, dev = device_now();   // taken under the lock, with the frame
    }
    plat_mutex_unlock(&al);
    // The driver's input stage is where it waits, when it wants to: outside the lock, so vk_antilag_present on the
    // present thread is not held up for the length of that wait.
    if (frame) {
        VkAntiLagPresentationInfoAMD p = {.sType = VK_STRUCTURE_TYPE_ANTI_LAG_PRESENTATION_INFO_AMD,
                                          .stage = VK_ANTI_LAG_STAGE_INPUT_AMD, .frameIndex = frame};
        update_on(fn, dev, 1, &p);
    }
}

// Before the device goes (vk_device.c vk_shutdown): the function and the state belong to that device, so they are
// let go; a device made again finds the extension and its feature anew (ready).
void vk_antilag_release(void) {
    plat_mutex_lock(&al);
    memset(&A, 0, sizeof A);
    plat_mutex_unlock(&al);
}

#else   // Vulkan headers older than VK_AMD_anti_lag: never available

int vk_antilag_available(void) { return 0; }
uint64_t vk_antilag_frame(void) { return 0; }
void vk_antilag_present(uint64_t frame) { (void)frame; }
void vk_antilag_input(void) {}
void vk_antilag_release(void) {}

#endif
