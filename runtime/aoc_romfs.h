// The add-on content RomFS (Donkey Kong Bananza's supplied DLC). The player's content is packed into an image and
// index the same way the game's own RomFS is (tools/dkb_pack_romfs.py, tools/romfs_index.py), but discovered as
// <exe folder>/aoc-<index>/ at start and parsed at runtime: romfs.c bakes its image hash and size in as generated
// constants and refuses a second mount name, so this is a reader of its own rather than a second mode of that one.
#pragma once
#include <stddef.h>
#include <stdint.h>

// Above the RomFS (0x1_...), host file (0x2_...) and directory (0x3_...) handle ranges.
#define AOC_HANDLE_BASE 0x400000000ull

// The add-on contents found at start, in ascending index order. AOC_MAX is what nn::aoc::ListAddOnContent can page.
enum { AOC_MAX = 32 };
uint32_t aoc_count(void);                                   // discovered add-on contents
uint32_t aoc_list(uint32_t *out, int offset, int count);    // writes up to count indices from offset; returns how many
const char *aoc_image_path(uint32_t index);                 // "<folder>/aoc-<index>/romfs.bin", for logs, or NULL

int aoc_owns(const char *path);                             // path is on the mounted add-on content
int aoc_handle(uint64_t h);
uint32_t aoc_mount(const char *name, uint32_t index);       // nn::fs::MountAddOnContent
uint32_t aoc_unmount(const char *name);
uint32_t aoc_open(uint64_t *out, const char *path, int mode);
uint32_t aoc_read(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size);
uint32_t aoc_size(int64_t *out, uint64_t h);
void aoc_close(uint64_t h);
uint32_t aoc_entry_type(int *out, const char *path);

int probe_aoc(void);   // aoc_probe.c: --probe-aoc mounts the real pack and reads it back
