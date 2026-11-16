// SPDX-License-Identifier: GPL-3.0-or-later
// The vulkan_driver setting (TOTK_VULKAN_DRIVER / DKB_VULKAN_DRIVER): a player's own Vulkan driver, such as a Mesa
// build outside the system's folders, for this game only. Linux only: elsewhere the setting does nothing.
#ifndef VK_DRIVER_ENV_H
#define VK_DRIVER_ENV_H

// Points the Vulkan loader at the ICD JSON file of the vulkan_driver setting by setting VK_DRIVER_FILES and
// VK_ICD_FILENAMES (the second for loaders before 1.3.207). The loader reads them when it first lists the drivers, so
// this runs from main right after mem_init (which reads the settings) and before SDL or vk_init is touched.
//   - empty setting, or not Linux: nothing changes, the system's drivers are used (0);
//   - VK_DRIVER_FILES or VK_ICD_FILENAMES already in the environment: the environment wins and nothing is set (0);
//   - the file exists: both variables become its absolute path, "~" expanded first (1);
//   - the file is missing: one log line, the system's drivers stay (0).
// Returns 1 only when it set the variables.
int vk_driver_env_apply(void);

#endif
