// --probe-aoc: mount the supplied add-on content next to the executable and read it back, so the reader is exercised on
// the real pack without the game. Every sampled file's size is compared with the index and the whole file hashed; with
// DKB_AOC_SOURCE=<romfs folder> (the folder tools/dkb_pack_romfs.py packed) the same bytes are read from the source and
// the two hashes compared, which is what says the pack holds the content rather than merely parsing.
//
// The index is parsed here rather than through aoc_romfs.c on purpose: the probe should not use the code it is checking
// for its own bookkeeping, only for the reads.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aoc_romfs.h"
#include "fs.h"
#include "host.h"
#include "platform/platform.h"

enum { SAMPLE = 64 };   // files to read, each hashed whole

static int hex_eq(const uint8_t a[32], const uint8_t b[32]) { return !memcmp(a, b, 32); }

static void hex32(const uint8_t d[32], char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

// The pack's row table, read straight from the index file.
typedef struct { uint64_t offset, size; uint32_t name, len; } Row;

int probe_aoc(void) {
    char exe[1600], path[1900];
    if (!plat_exe_path(exe, sizeof exe)) {
        printf("probe-aoc: no executable path\n");
        return 1;
    }
    char *cut = plat_last_sep(exe);
    if (cut) *cut = 0;

    uint32_t index = 0;
    int found = 0;
    for (uint32_t i = 0; i < 32 && !found; i++) {
        snprintf(path, sizeof path, "%s/aoc-%u/aoc-index.bin", exe, i);
        if (plat_file_exists(path)) index = i, found = 1;
    }
    if (!found) {
        printf("probe-aoc: skipped: no aoc-<index>/ next to the executable (%s)\n", exe);
        return 0;
    }

    snprintf(path, sizeof path, "%s/aoc-%u/aoc-index.bin", exe, index);
    uint64_t size = 0;
    uint8_t *idx = plat_file_read(path, &size);
    if (!idx) {
        printf("probe-aoc: FAIL: cannot read %s\n", path);
        return 1;
    }
    if (size < 24 || memcmp(idx, "TOTKRFS1", 8)) {
        printf("probe-aoc: FAIL: %s is not a TOTKRFS1 index\n", path);
        return 1;
    }
    uint64_t rows, strings;
    memcpy(&rows, idx + 8, 8);
    memcpy(&strings, idx + 16, 8);
    if (!rows || strings != 24 + rows * 24 || strings >= size) {
        printf("probe-aoc: FAIL: index layout: %llu rows, strings at %llu of %llu\n", (unsigned long long)rows,
               (unsigned long long)strings, (unsigned long long)size);
        return 1;
    }

    const char *src = getenv("DKB_AOC_SOURCE");   // env, not cfg: this is probe-only, with no dkb.ini row
    uint32_t rc = aoc_mount("probeaoc", index);
    if (rc) {
        printf("probe-aoc: FAIL: aoc_mount(%u) = %#x\n", index, rc);
        free(idx);
        return 1;
    }

    uint64_t want = rows < SAMPLE ? rows : SAMPLE;
    uint64_t checked = 0, compared = 0, bytes = 0;
    int failures = 0;
    for (uint64_t k = 0; k < want; k++) {
        Row r;
        memcpy(&r, idx + 24 + k * 24, sizeof r);
        const char *rel = (const char *)idx + strings + r.name;
        uint64_t h;
        snprintf(path, sizeof path, "probeaoc:/%s", rel);
        uint32_t o = aoc_open(&h, path, FS_OPEN_READ);
        if (o) {
            printf("  open failed %#x: %s\n", o, rel);
            failures++;
            continue;
        }
        int64_t got = -1;
        aoc_size(&got, h);
        if ((uint64_t)got != r.size) {
            printf("  size %lld != index %llu: %s\n", (long long)got, (unsigned long long)r.size, rel);
            failures++;
        }
        uint8_t *buf = malloc(r.size ? r.size : 1);
        if (!buf) {
            printf("  out of memory for %llu bytes: %s\n", (unsigned long long)r.size, rel);
            aoc_close(h);
            failures++;
            continue;
        }
        uint64_t n = 0;
        uint32_t rd = aoc_read(&n, h, 0, buf, r.size);
        aoc_close(h);
        if (rd) {
            printf("  read failed %#x: %s\n", rd, rel);
            free(buf);
            failures++;
            continue;
        }
        checked++;
        bytes += n;
        uint8_t mine[32];
        rt_sha256(buf, (size_t)n, mine);
        free(buf);
        if (src && *src) {
            char sp[2100];
            snprintf(sp, sizeof sp, "%s/%s", src, rel);
            uint64_t ssize = 0;
            uint8_t *sbuf = plat_file_read(sp, &ssize);
            if (!sbuf) {
                printf("  source missing: %s\n", sp);
                failures++;
                continue;
            }
            uint8_t theirs[32];
            rt_sha256(sbuf, (size_t)(ssize < n ? ssize : n), theirs);
            free(sbuf);
            if (ssize != n || !hex_eq(mine, theirs)) {
                char a[65], b[65];
                hex32(mine, a);
                hex32(theirs, b);
                printf("  bytes differ: %s\n    pack   %s\n    source %s\n", rel, a, b);
                failures++;
                continue;
            }
            compared++;
        }
    }
    aoc_unmount("probeaoc");   // undo the mount; the probe may run in a process that goes on
    free(idx);

    printf("probe-aoc: content %u, %llu files in the index, %llu read (%llu bytes, each hashed whole), %llu compared "
           "with the source folder\n", index, (unsigned long long)rows, (unsigned long long)checked,
           (unsigned long long)bytes, (unsigned long long)compared);
    if (failures) {
        printf("probe-aoc: FAILED (%d)\n", failures);
        return 1;
    }
    if (src && *src && compared != checked) {
        printf("probe-aoc: FAILED: %llu of %llu files reached the source\n", (unsigned long long)compared,
               (unsigned long long)checked);
        return 1;
    }
    printf("probe-aoc: passed%s\n", src && *src ? " (bytes match the source folder)" : " (mount and read only)");
    return 0;
}
