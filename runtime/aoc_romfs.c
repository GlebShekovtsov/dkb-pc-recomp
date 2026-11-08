// Adapted from MacNativeGames games/totk/runtime/romfs.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// The add-on content RomFS: the supplied DLC, packed into an image and index like the game's own RomFS
// (tools/dkb_pack_romfs.py, tools/romfs_index.py). It is discovered as <exe folder>/aoc-<index>/ at first use and its
// index is parsed here at runtime rather than baked in, because romfs.c holds the base image's hash and size as
// generated constants and refuses a second mount name; this reader keeps that path untouched.
//
// One add-on content is mounted at a time: the game mounts what nn::aoc reports, and a second name is answered rather
// than silently replacing the first, so a wrong assumption shows up as a mount failure instead of wrong content.
//
// Locking: `lock` guards the mount (name, index, image, handle table). Readers hold it shared across the index lookup
// and the positional read, and mount and unmount take it exclusively, so an unmount waits for reads in flight instead
// of freeing the index and closing the image under them. That is simpler than refusing to unmount while handles are
// open (the game may unmount with handles it never closed) and costs nothing: shared holders do not exclude each other.
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aoc_romfs.h"
#include "fs.h"
#include "host.h"
#include "platform/platform.h"
#include "romfs_path.h"

typedef struct { uint64_t offset, size; uint32_t name, len; } Row;   // a TOTKRFS1 row, as tools/romfs_index.py writes it

enum { AOC_SLOTS = 256 };

static char aoc_root[1800];             // the exe's folder: where aoc-<index> directories are looked for
static uint32_t aoc_ids[AOC_MAX];
static int aoc_n;                       // written once, by scan_once, before scan() returns to anyone
static PlatMutex lock = PLAT_MUTEX_INIT;

// The mounted add-on content.
static char mount_name[32];
static uint32_t mount_index;
static uint8_t *idx;                    // the index bytes, read once at mount
static size_t idx_size;
static uint64_t rows, strings_off;
static PlatFile *image;
static uint64_t handles[AOC_SLOTS];     // index row + 1; 0: free

static Row row(uint64_t i) {
    Row r;
    memcpy(&r, idx + 24 + i * 24, sizeof r);
    return r;
}

static const char *row_name(Row r) { return (const char *)idx + strings_off + r.name; }

// The first row whose path is not below prefix, and whether that row starts with it. The caller holds the lock.
static uint64_t lower_bound(const char *prefix, size_t n, int *matches) {
    uint64_t lo = 0, hi = rows;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (strncmp(row_name(row(mid)), prefix, n) < 0) lo = mid + 1;
        else hi = mid;
    }
    *matches = lo < rows && !strncmp(row_name(row(lo)), prefix, n) && row(lo).len > n;
    return lo;
}

// The caller holds the lock.
static uint64_t find(const char *path) {
    uint64_t lo = 0, hi = rows;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        int c = strcmp(path, row_name(row(mid)));
        if (!c) return mid;
        if (c < 0) hi = mid;
        else lo = mid + 1;
    }
    return UINT64_MAX;
}

static void scan_once(void *param) {
    (void)param;
    char exe[1600];
    if (!plat_exe_path(exe, sizeof exe)) {
        rt_log("aoc: no executable path; add-on content unavailable");
        return;
    }
    char *cut = plat_last_sep(exe);
    if (!cut) return;
    *cut = 0;
    snprintf(aoc_root, sizeof aoc_root, "%s", exe);
    char path[1900];
    for (uint32_t i = 0; i < AOC_MAX && aoc_n < AOC_MAX; i++) {
        snprintf(path, sizeof path, "%s/aoc-%u/aoc-index.bin", aoc_root, i);
        if (!plat_file_exists(path)) continue;
        snprintf(path, sizeof path, "%s/aoc-%u/romfs.bin", aoc_root, i);
        if (!plat_file_exists(path)) continue;
        aoc_ids[aoc_n++] = i;
    }
    rt_log("aoc: %d add-on content%s next to the executable", aoc_n, aoc_n == 1 ? "" : "s");
}

// Looks for the aoc-<index> folders next to the executable, once; every caller returns after the scan has finished.
static void scan(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, scan_once, NULL);
}

uint32_t aoc_count(void) {
    scan();
    return (uint32_t)aoc_n;
}

uint32_t aoc_list(uint32_t *out, int offset, int count) {
    scan();
    if (!out || offset < 0 || count < 0) return 0;
    int n = 0;
    for (int i = offset; i < aoc_n && n < count; i++) out[n++] = aoc_ids[i];
    return (uint32_t)n;
}

const char *aoc_image_path(uint32_t index) {
    static char path[1900];
    scan();
    for (int i = 0; i < aoc_n; i++)
        if (aoc_ids[i] == index) {
            snprintf(path, sizeof path, "%s/aoc-%u/romfs.bin", aoc_root, index);
            return path;
        }
    return NULL;
}

// The caller holds the lock, shared or exclusive.
static int owns_locked(const char *path) {
    size_t n = strlen(mount_name);
    return n && path && !strncmp(path, mount_name, n) && path[n] == ':';
}

int aoc_owns(const char *path) {
    plat_mutex_lock_shared(&lock);
    int r = owns_locked(path);
    plat_mutex_unlock_shared(&lock);
    return r;
}

int aoc_handle(uint64_t h) { return h > AOC_HANDLE_BASE && h < AOC_HANDLE_BASE + AOC_SLOTS; }

// Drops a half-built mount and releases the lock (aoc_mount holds it) before stopping the game, so nothing is left
// locked or half mounted behind the stop.
static void mount_fail(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void mount_fail(const char *fmt, ...) {
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    if (image) plat_file_close(image);
    image = NULL;
    free(idx);
    idx = NULL;
    rows = strings_off = 0;
    plat_mutex_unlock(&lock);
    rt_fatal("aoc: %s", msg);
}
#define mneed(c, m) do { if (!(c)) mount_fail("%s", m); } while (0)
#define mfatal(...) mount_fail(__VA_ARGS__)

uint32_t aoc_mount(const char *name, uint32_t index) {
    if (!name || !*name || !romfs_path_valid_relative(name) || strlen(name) >= sizeof mount_name) return FS_R_INVALID;
    scan();
    int present = 0;
    for (int i = 0; i < aoc_n; i++)
        if (aoc_ids[i] == index) present = 1;
    if (!present) return FS_R_PATH_NOT_FOUND;   // no such add-on content on this machine
    plat_mutex_lock(&lock);
    if (*mount_name) {
        int same = mount_index == index && !strcmp(mount_name, name);
        plat_mutex_unlock(&lock);
        return same ? 0 : FS_R_ALREADY_EXISTS;
    }
    char path[1900];
    snprintf(path, sizeof path, "%s/aoc-%u/aoc-index.bin", aoc_root, index);
    uint64_t size = 0;
    idx = plat_file_read(path, &size);
    if (!idx) mfatal("add-on content %u has no readable index %s", index, path);
    mneed(size >= 24 && size < (64 << 20), "index size");
    idx_size = (size_t)size;
    mneed(!memcmp(idx, "TOTKRFS1", 8), "index format");
    memcpy(&rows, idx + 8, 8);
    memcpy(&strings_off, idx + 16, 8);
    mneed(rows < 1000000 && strings_off == 24 + rows * 24 && strings_off < idx_size, "index layout");
    snprintf(path, sizeof path, "%s/aoc-%u/romfs.bin", aoc_root, index);
    image = plat_file_open(path);
    if (!image) mfatal("add-on content %u has no readable image %s", index, path);
    PlatFileId id;
    mneed(plat_file_id(image, &id), "image size");
    const char *previous = NULL;
    for (uint64_t i = 0; i < rows; i++) {
        Row r = row(i);
        mneed(r.offset <= id.size && r.size <= id.size - r.offset &&
                 r.name < idx_size - strings_off && r.len > 1 && r.len <= idx_size - strings_off - r.name &&
                 row_name(r)[r.len - 1] == 0 && strlen(row_name(r)) == r.len - 1,
             "index row");
        // find and lower_bound binary-search the rows, so the paths must be strictly ascending (romfs.c load_index)
        mneed(!previous || strcmp(previous, row_name(r)) < 0, "sorted paths");
        previous = row_name(r);
    }
    memset(handles, 0, sizeof handles);
    mount_index = index;
    snprintf(mount_name, sizeof mount_name, "%s", name);
    plat_mutex_unlock(&lock);
    rt_log("fs mount %s: -> add-on content %u, %llu files, %s", name, index, (unsigned long long)rows, path);
    return 0;
}

uint32_t aoc_unmount(const char *name) {
    if (!name) return FS_R_INVALID;
    plat_mutex_lock(&lock);
    if (!*mount_name || strcmp(name, mount_name)) {
        plat_mutex_unlock(&lock);
        return FS_R_INVALID;
    }
    if (image) plat_file_close(image);
    image = NULL;
    free(idx);
    idx = NULL;
    rows = strings_off = 0;
    *mount_name = 0;
    memset(handles, 0, sizeof handles);
    plat_mutex_unlock(&lock);
    rt_log("fs unmount %s: add-on content", name);
    return 0;
}

uint32_t aoc_open(uint64_t *out, const char *path, int mode) {
    if (!out) return FS_R_INVALID;
    plat_mutex_lock(&lock);   // exclusive: the lookup and the handle slot belong to one mount
    if (!owns_locked(path)) {
        plat_mutex_unlock(&lock);
        return FS_R_INVALID;
    }
    if (mode != FS_OPEN_READ) {   // add-on content is read-only
        plat_mutex_unlock(&lock);
        return FS_R_UNSUPPORTED_OPEN;
    }
    const char *p = romfs_path_relative(mount_name, path);
    uint64_t i = p ? find(p) : UINT64_MAX;
    if (i == UINT64_MAX) {
        plat_mutex_unlock(&lock);
        return FS_R_PATH_NOT_FOUND;
    }
    for (int j = 1; j < AOC_SLOTS; j++)
        if (!handles[j]) {
            handles[j] = i + 1;
            *out = AOC_HANDLE_BASE + (uint64_t)j;
            plat_mutex_unlock(&lock);
            return 0;
        }
    plat_mutex_unlock(&lock);
    rt_fatal("aoc: file handle table full (%d)", AOC_SLOTS);
    return FS_R_INVALID;
}

// The row a handle names, or 0. The caller holds the lock.
static int source_locked(uint64_t h, Row *r) {
    if (!aoc_handle(h)) return 0;
    uint64_t i = handles[h - AOC_HANDLE_BASE];
    if (i && idx) *r = row(i - 1);
    return i != 0 && idx != NULL;
}

uint32_t aoc_read(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size) {
    Row r;
    if (!out || offset < 0) return FS_R_INVALID;
    plat_mutex_lock_shared(&lock);   // held through the reads: an unmount closes the image
    if (!source_locked(h, &r) || (uint64_t)offset > r.size || (!buffer && size)) {
        plat_mutex_unlock_shared(&lock);
        return FS_R_INVALID;
    }
    uint64_t n = size < r.size - (uint64_t)offset ? size : r.size - (uint64_t)offset;
    uint64_t done = 0;
    while (done < n) {
        uint32_t chunk = n - done > (1u << 20) ? (1u << 20) : (uint32_t)(n - done);
        mem_thaw((uintptr_t)buffer + done, chunk);
        int64_t got = plat_file_pread(image, (uint8_t *)buffer + done, chunk, r.offset + (uint64_t)offset + done);
        if (got <= 0) rt_fatal("aoc: read failure at %#llx", (unsigned long long)(r.offset + offset + done));
        done += (uint64_t)got;
    }
    plat_mutex_unlock_shared(&lock);
    *out = done;
    return 0;
}

uint32_t aoc_size(int64_t *out, uint64_t h) {
    Row r;
    if (!out) return FS_R_INVALID;
    plat_mutex_lock_shared(&lock);
    int ok = source_locked(h, &r);
    plat_mutex_unlock_shared(&lock);
    if (!ok) return FS_R_INVALID;
    *out = (int64_t)r.size;
    return 0;
}

void aoc_close(uint64_t h) {
    if (!aoc_handle(h)) rt_fatal("aoc: close of invalid file handle %#llx", (unsigned long long)h);
    plat_mutex_lock(&lock);
    handles[h - AOC_HANDLE_BASE] = 0;
    plat_mutex_unlock(&lock);
}

uint32_t aoc_entry_type(int *out, const char *path) {
    if (!out) return FS_R_INVALID;
    plat_mutex_lock_shared(&lock);
    uint32_t r = FS_R_INVALID;
    if (owns_locked(path)) {
        const char *p = romfs_path_relative(mount_name, path);
        char dir[4200];
        int matches = 0;
        if (!p) {
            r = FS_R_PATH_NOT_FOUND;
        } else if (find(p) != UINT64_MAX) {
            *out = FS_ENTRY_FILE;
            r = 0;
        } else if (snprintf(dir, sizeof dir, "%s/", p) < (int)sizeof dir) {
            lower_bound(dir, strlen(dir), &matches);
            r = matches ? 0 : FS_R_PATH_NOT_FOUND;
            if (matches) *out = FS_ENTRY_DIRECTORY;
        }
    }
    plat_mutex_unlock_shared(&lock);
    return r;
}
