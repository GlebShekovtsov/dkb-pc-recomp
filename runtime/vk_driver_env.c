// SPDX-License-Identifier: GPL-3.0-or-later
// The vulkan_driver setting: see vk_driver_env.h.
#include "vk_driver_env.h"

#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

#ifndef _WIN32
// path with a leading "~" or "~/" replaced by the user's home folder (HOME); unchanged when there is no such prefix or
// no home. 0 when the result does not fit cap.
static int expand_home(const char *path, char *out, size_t cap) {
    char home[1024];
    if (path[0] == '~' && (!path[1] || path[1] == '/') && plat_getenv("HOME", home, sizeof home)) {
        int n = snprintf(out, cap, "%s%s", home, path + 1);
        return n > 0 && (size_t)n < cap;
    }
    int n = snprintf(out, cap, "%s", path);
    return n > 0 && (size_t)n < cap;
}
#endif

int vk_driver_env_apply(void) {
#ifdef _WIN32
    return 0;   // the setting is for Mesa builds on Linux; Windows drivers come from the registry
#else
    const char *want = cfg_value("TOTK_VULKAN_DRIVER");
    if (!*want) return 0;
    char probe[4096];   // plat_getenv gives 0 for a value that does not fit, so keep it roomy
    if (plat_getenv("VK_DRIVER_FILES", probe, sizeof probe) || plat_getenv("VK_ICD_FILENAMES", probe, sizeof probe)) {
        rt_log("vk: VK_DRIVER_FILES or VK_ICD_FILENAMES is set in the environment, it wins over vulkan_driver");
        return 0;
    }
    char path[1024], full[1024];
    if (!expand_home(want, path, sizeof path) || !plat_file_exists(path) || !plat_full_path(path, full, sizeof full)) {
        rt_log("vk: vulkan_driver '%s' is not a file, the system's Vulkan drivers stay", want);
        return 0;
    }
    plat_setenv("VK_DRIVER_FILES", full);
    plat_setenv("VK_ICD_FILENAMES", full);
    rt_log("vk: Vulkan driver from vulkan_driver: %s", full);
    return 1;
#endif
}
