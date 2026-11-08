// Standalone check of runtime/aoc_romfs.c against SYNTHETIC add-on content (no game data): packs of a few made-up files
// with TOTKRFS1 indexes written here, placed as aoc-<index>/ folders next to this executable, which is where the reader
// looks. rt_fatal is replaced by a longjmp so the rejected-index and full-table cases can be observed.
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aoc_romfs.h"
#include "dispatch.h"
#include "fs.h"
#include "host.h"
#include "platform/platform.h"

static jmp_buf fatal_jmp;
static int fatal_armed;
static char fatal_msg[256];

void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(fatal_msg, sizeof fatal_msg, fmt, ap);
    va_end(ap);
    if (!fatal_armed) {
        fprintf(stderr, "unexpected rt_fatal: %s\n", fatal_msg);
        exit(2);
    }
    fatal_armed = 0;
    longjmp(fatal_jmp, 1);
}
void rt_log(const char *fmt, ...) { (void)fmt; }
void mem_thaw(uint64_t a, uint64_t n) { (void)a, (void)n; }

static int failures;
#define CHECK(c) do { if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

// Runs a mount that must stop the game with a message containing `why`.
#define EXPECT_FATAL(call, why) do { \
        fatal_armed = 1; \
        if (!setjmp(fatal_jmp)) { (void)(call); fatal_armed = 0; printf("FAIL %s:%d: no fatal\n", __FILE__, __LINE__); failures++; } \
        else CHECK(strstr(fatal_msg, why) != NULL); \
    } while (0)

typedef struct { uint64_t offset, size; uint32_t name, len; } Row;

static char root[1700];

static void put_pack(unsigned index, const char *const *names, const uint64_t *offs, const uint64_t *sizes, int n,
                     const void *image, size_t image_size, int bad_row) {
    char dir[1800], path[1900];
    snprintf(dir, sizeof dir, "%s/aoc-%u", root, index);
    plat_file_mkdir_p(dir);
    snprintf(path, sizeof path, "%s/romfs.bin", dir);
    CHECK(plat_file_write(path, image, image_size));
    size_t strings = 0;
    for (int i = 0; i < n; i++) strings += strlen(names[i]) + 1;
    size_t total = 24 + (size_t)n * 24 + strings;
    uint8_t *idx = calloc(1, total);
    uint64_t rows = (uint64_t)n, so = 24 + (uint64_t)n * 24;
    memcpy(idx, "TOTKRFS1", 8);
    memcpy(idx + 8, &rows, 8);
    memcpy(idx + 16, &so, 8);
    uint32_t at = 0;
    for (int i = 0; i < n; i++) {
        Row r = {offs[i], sizes[i], at, (uint32_t)strlen(names[i]) + 1};
        if (bad_row && i == 1) r.size = image_size + 1;   // runs past the image
        memcpy(idx + 24 + (size_t)i * 24, &r, sizeof r);
        memcpy(idx + so + at, names[i], r.len);
        at += r.len;
    }
    snprintf(path, sizeof path, "%s/aoc-index.bin", dir);
    CHECK(plat_file_write(path, idx, total));
    free(idx);
}

int main(void) {
    char exe[1600];
    if (!plat_exe_path(exe, sizeof exe)) return 2;
    *plat_last_sep(exe) = 0;
    snprintf(root, sizeof root, "%s", exe);

    uint8_t image[200];
    for (int i = 0; i < (int)sizeof image; i++) image[i] = (uint8_t)(i * 7 + 3);
    const char *good[] = {"a.txt", "dir/b.bin", "dir/sub/c"};
    const char *unsorted[] = {"dir/b.bin", "a.txt"};
    const char *dup[] = {"a.txt", "a.txt"};
    uint64_t offs[] = {0, 10, 150}, sizes[] = {5, 100, 3}, offs2[] = {10, 0}, sizes2[] = {100, 5};
    put_pack(0, good, offs, sizes, 3, image, sizeof image, 0);
    put_pack(1, good, offs, sizes, 3, image, sizeof image, 1);      // malformed row
    put_pack(2, unsorted, offs2, sizes2, 2, image, sizeof image, 0);   // unsorted
    put_pack(3, dup, offs, sizes, 2, image, sizeof image, 0);       // not strictly ascending

    // ListAddOnContent: four contents, paged.
    uint32_t ids[8] = {0};
    CHECK(aoc_count() == 4);
    CHECK(aoc_list(ids, 0, 8) == 4 && ids[0] == 0 && ids[3] == 3);
    CHECK(aoc_list(ids, 1, 2) == 2 && ids[0] == 1 && ids[1] == 2);   // an offset and a count
    CHECK(aoc_list(ids, 3, 8) == 1 && ids[0] == 3);
    CHECK(aoc_list(ids, 4, 8) == 0 && aoc_list(ids, -1, 8) == 0 && aoc_list(ids, 0, -1) == 0);
    CHECK(aoc_image_path(0) && !aoc_image_path(9));

    // Rejected indexes stop the mount and leave nothing mounted.
    EXPECT_FATAL(aoc_mount("bad", 1), "index row");
    CHECK(!aoc_owns("bad:/a.txt"));
    aoc_unmount("bad");
    EXPECT_FATAL(aoc_mount("bad", 2), "sorted paths");
    aoc_unmount("bad");
    EXPECT_FATAL(aoc_mount("bad", 3), "sorted paths");
    aoc_unmount("bad");
    CHECK(aoc_mount("x", 9) == FS_R_PATH_NOT_FOUND);
    CHECK(aoc_mount("a/../b", 0) == FS_R_INVALID);

    // The good pack.
    CHECK(aoc_mount("aoc", 0) == 0);
    CHECK(aoc_mount("aoc", 0) == 0);                      // the same mount again is answered
    CHECK(aoc_mount("other", 0) == FS_R_ALREADY_EXISTS);  // a second name is refused
    CHECK(aoc_owns("aoc:/a.txt") && !aoc_owns("rom:/a.txt"));

    int type = -1;
    CHECK(aoc_entry_type(&type, "aoc:/a.txt") == 0 && type == FS_ENTRY_FILE);
    CHECK(aoc_entry_type(&type, "aoc:/dir") == 0 && type == FS_ENTRY_DIRECTORY);
    CHECK(aoc_entry_type(&type, "aoc:/dir/sub") == 0 && type == FS_ENTRY_DIRECTORY);
    CHECK(aoc_entry_type(&type, "aoc:/dir/b") == FS_R_PATH_NOT_FOUND);   // a prefix of a file name is not a directory
    CHECK(aoc_entry_type(&type, "aoc:/nope") == FS_R_PATH_NOT_FOUND);
    CHECK(aoc_entry_type(&type, "aoc:/") == FS_R_PATH_NOT_FOUND);        // the root: the reader does not answer it
    CHECK(aoc_entry_type(&type, "rom:/a.txt") == FS_R_INVALID);

    uint64_t h = 0, n = 0;
    CHECK(aoc_open(&h, "aoc:/dir/b.bin", FS_OPEN_WRITE) == FS_R_UNSUPPORTED_OPEN);     // read-only
    CHECK(aoc_open(&h, "aoc:/dir/b.bin", FS_OPEN_READ | FS_OPEN_WRITE) == FS_R_UNSUPPORTED_OPEN);
    CHECK(aoc_open(&h, "aoc:/nope", FS_OPEN_READ) == FS_R_PATH_NOT_FOUND);
    CHECK(aoc_open(&h, "aoc:/dir", FS_OPEN_READ) == FS_R_PATH_NOT_FOUND);              // a directory is not a file
    CHECK(aoc_open(&h, "aoc:/", FS_OPEN_READ) == FS_R_PATH_NOT_FOUND);
    CHECK(aoc_open(&h, "rom:/a.txt", FS_OPEN_READ) == FS_R_INVALID);
    CHECK(aoc_open(&h, "aoc:/dir/b.bin", FS_OPEN_READ) == 0 && aoc_handle(h));

    int64_t size = 0;
    uint8_t buf[256];
    CHECK(aoc_size(&size, h) == 0 && size == 100);
    CHECK(aoc_read(&n, h, 0, buf, 100) == 0 && n == 100 && !memcmp(buf, image + 10, 100));
    CHECK(aoc_read(&n, h, 40, buf, 20) == 0 && n == 20 && !memcmp(buf, image + 50, 20));
    CHECK(aoc_read(&n, h, 90, buf, 50) == 0 && n == 10 && !memcmp(buf, image + 100, 10));   // clipped at the end
    CHECK(aoc_read(&n, h, 100, buf, 8) == 0 && n == 0);                                      // at EOF
    CHECK(aoc_read(&n, h, 101, buf, 8) == FS_R_INVALID);                                     // past EOF
    CHECK(aoc_read(&n, h, -1, buf, 8) == FS_R_INVALID);
    CHECK(aoc_read(&n, h, 0, NULL, 8) == FS_R_INVALID);
    CHECK(aoc_read(&n, AOC_HANDLE_BASE + 99, 0, buf, 8) == FS_R_INVALID);                  // a handle never opened
    aoc_close(h);
    CHECK(aoc_read(&n, h, 0, buf, 8) == FS_R_INVALID);                                      // closed

    // A full handle table stops the game; closing handles frees slots again.
    uint64_t hs[300];
    volatile int opened = 0;
    fatal_armed = 1;
    if (!setjmp(fatal_jmp))
        for (; opened < 300; opened++) CHECK(aoc_open(&hs[opened], "aoc:/a.txt", FS_OPEN_READ) == 0);
    CHECK(opened == 255 && strstr(fatal_msg, "handle table full"));
    for (int i = 0; i < opened; i++) aoc_close(hs[i]);
    CHECK(aoc_open(&h, "aoc:/a.txt", FS_OPEN_READ) == 0);

    // Unmount drops the open handle and the mount; a remount works and starts with an empty table.
    CHECK(aoc_unmount("other") == FS_R_INVALID);
    CHECK(aoc_unmount("aoc") == 0);
    CHECK(!aoc_owns("aoc:/a.txt") && aoc_open(&n, "aoc:/a.txt", FS_OPEN_READ) == FS_R_INVALID);
    CHECK(aoc_read(&n, h, 0, buf, 5) == FS_R_INVALID && aoc_size(&size, h) == FS_R_INVALID);
    CHECK(aoc_unmount("aoc") == FS_R_INVALID);
    CHECK(aoc_mount("again", 0) == 0);
    CHECK(aoc_read(&n, h, 0, buf, 5) == FS_R_INVALID);
    CHECK(aoc_open(&h, "again:/a.txt", FS_OPEN_READ) == 0 && aoc_read(&n, h, 0, buf, 5) == 0 && n == 5 &&
          !memcmp(buf, image, 5));
    aoc_close(h);
    CHECK(aoc_unmount("again") == 0);

    printf(failures ? "aoc_romfs: %d failure(s)\n" : "aoc_romfs: all passed\n", failures);
    return failures != 0;
}
