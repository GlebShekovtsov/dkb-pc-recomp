#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "host.h"

static RtHostEntry display_version;
static RtHostEntry count_content;
static RtHostEntry list_content;
static RtHostEntry network[4];
static RtHostEntry bcat_init, bcat_mount;
static RtHostEntry stream_memory;
static RtHostEntry cached_account;
static const char *network_symbols[] = {
    "_ZN2nn4nifm10InitializeEv",
    "_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv",
    "_ZN2nn4nifm18IsNetworkAvailableEv",
    "_ZN2nn4nifm31HandleNetworkRequestErrorResultEv",
};
#define DISPLAY_VERSION_SYMBOL "_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE"
#define COUNT_CONTENT_SYMBOL "_ZN2nn3aoc17CountAddOnContentEv"
#define LIST_CONTENT_SYMBOL "_ZN2nn3aoc16ListAddOnContentEPiii"
#undef HOST_OVERRIDE
#define HOST_OVERRIDE(sym, fn) \
    static void __attribute__((constructor)) RT_CAT(register_system_, __COUNTER__)(void) { \
        if (!strcmp(sym, DISPLAY_VERSION_SYMBOL)) display_version = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, COUNT_CONTENT_SYMBOL)) count_content = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, LIST_CONTENT_SYMBOL)) list_content = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4nifm10InitializeEv")) network[0] = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4nifm27SubmitNetworkRequestAndWaitEv")) network[1] = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4nifm18IsNetworkAvailableEv")) network[2] = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4nifm31HandleNetworkRequestErrorResultEv")) network[3] = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4bcat10InitializeEv")) bcat_init = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn4bcat25MountDeliveryCacheStorageEv")) bcat_mount = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn10streamplay21GetRequiredMemorySizeENS0_14StreamPlayTypeENS0_13StreamingModeEi")) stream_memory = (RtHostEntry){sym, fn}; \
        if (!strcmp(sym, "_ZN2nn7account29LoadCachedNintendoAccountInfoEPNS0_25CachedNintendoAccountInfoERKNS0_10UserHandleEPvm")) cached_account = (RtHostEntry){sym, fn}; \
    }
#undef HOST_OBSERVER
#define HOST_OBSERVER(sym, fn)
#include "ov_system.c"

void rt_log(const char *fmt, ...) { (void)fmt; }
void rt_fatal(const char *fmt, ...) { (void)fmt; abort(); }
const int rt_game_version;
const char rt_game_target[] = "dkb458752";   // update 3.2.0: GetDisplayVersion still reports "unknown" for it
const char rt_game_display[] = "3.2.0";
const char *cfg_value(const char *key) { (void)key; abort(); }
const char *cfg_get(const char *key) { (void)key; abort(); }
const char *plat_user_locale(void) { abort(); }
uint64_t rt_ticks(void) { return 0; }
const RtConfig *rt_config(void) { abort(); }
void plat_mutex_lock(PlatMutex *m) { (void)m; abort(); }
void plat_mutex_unlock(PlatMutex *m) { (void)m; abort(); }
void plat_mutex_lock_shared(PlatMutex *m) { assert(m == &account_lock); }
void plat_mutex_unlock_shared(PlatMutex *m) { assert(m == &account_lock); }
void plat_cond_broadcast(PlatCond *cv) { (void)cv; abort(); }
void plat_file_mkdir_p(const char *path) { (void)path; abort(); }
int plat_dir_exists(const char *path) { (void)path; abort(); }
int64_t plat_file_size(const char *path) { (void)path; abort(); }
uint8_t *plat_file_read(const char *path, uint64_t *size) { (void)path; (void)size; abort(); }
int plat_random(void *out, size_t size) { (void)out; (void)size; abort(); }
int plat_file_write_atomic(const char *path, const void *data, size_t size) {
    (void)path; (void)data; (void)size; abort();
}
int plat_exe_path(char *out, size_t cap) { (void)out; (void)cap; abort(); }
int plat_file_exists(const char *path) { (void)path; abort(); }

int main(void) {
    if (!cached_account.fn) {
        fputs("missing offline cached Nintendo Account binding\n", stderr);
        return 1;
    }
    uint8_t cached[0x80], buffer[0x800];
    memset(cached, 0xa5, sizeof cached);
    memset(buffer, 0x5a, sizeof buffer);
    local = (Uid){7, 9};
    tokens[0] = 1;
    UserHandle handle = {local, 1};
    Cpu account = {0};
    account.x[0] = (uintptr_t)cached;
    account.x[1] = (uintptr_t)&handle;
    account.x[2] = (uintptr_t)buffer;
    account.x[3] = sizeof buffer;
    for (int i = 19; i < 31; i++) account.x[i] = 0xa000 + i;
    cached_account.fn(&account);
    assert(account.x[0] == ((200u << 9) | 124u));
    handle.token = 2;
    account.x[0] = (uintptr_t)cached;
    cached_account.fn(&account);
    assert(account.x[0] == ACCOUNT_R_INVALID);
    account.x[1] = 0;
    cached_account.fn(&account);
    assert(account.x[0] == ACCOUNT_R_INVALID);
    for (size_t i = 0; i < sizeof cached; i++) assert(cached[i] == 0xa5);
    for (size_t i = 0; i < sizeof buffer; i++) assert(buffer[i] == 0x5a);
    for (int i = 19; i < 31; i++) assert(account.x[i] == (uint64_t)(0xa000 + i));
    if (!stream_memory.fn) {
        fputs("missing offline streamplay memory query\n", stderr);
        return 1;
    }
    Cpu stream = {0};
    stream.x[0] = 1;
    stream.x[2] = 3;
    stream_memory.fn(&stream);
    assert(stream.x[0] == 0);
    if (!display_version.fn) {
        fputs("missing GetDisplayVersion host binding\n", stderr);
        return 1;
    }
    struct { uint8_t before; char value[16]; uint8_t after; } out;
    memset(&out, 0xa5, sizeof out);
    Cpu c = {0};
    c.x[0] = (uintptr_t)out.value;
    for (int i = 1; i < 31; i++) c.x[i] = 0x1900 + i;
    display_version.fn(&c);
    assert(!strcmp(out.value, "unknown"));
    for (size_t i = sizeof "unknown"; i < sizeof out.value; i++) assert(out.value[i] == 0);
    assert(out.before == 0xa5 && out.after == 0xa5);
    for (int i = 1; i < 31; i++) assert(c.x[i] == (uint64_t)(0x1900 + i));
    if (!count_content.fn) {
        fputs("missing CountAddOnContent host binding\n", stderr);
        return 1;
    }
    count_content.fn(&c);
    assert(c.x[0] == 0);
    assert(list_content.fn);
    int index = -1;
    c.x[0] = (uintptr_t)&index;
    c.x[1] = 0;
    c.x[2] = 1;
    list_content.fn(&c);
    assert(c.x[0] == 0 && index == -1);
    for (int i = 0; i < 4; i++) {
        if (!network[i].fn) {
            fprintf(stderr, "missing offline network binding: %s\n", network_symbols[i]);
            return 1;
        }
        network[i].fn(&c);
        if (i >= 2) assert(c.x[0] == 0);
        for (int r = 19; r < 31; r++) assert(c.x[r] == (uint64_t)(0x1900 + r));
    }
    if (!bcat_init.fn || !bcat_mount.fn) {
        fputs("missing offline BCAT bindings\n", stderr);
        return 1;
    }
    bcat_init.fn(&c);
    bcat_mount.fn(&c);
    assert(c.x[0] == 0x47a);
    puts("system services: display version, empty AOC and offline network/BCAT bindings passed");
    return 0;
}

// aoc_romfs.c is not part of this harness: the reader's surface is stubbed at its boundary, since these overrides are
// what is under test here. The reader itself is covered by --probe-aoc.
uint32_t aoc_count(void) { return 0; }
uint32_t aoc_list(uint32_t *out, int offset, int count) { (void)out, (void)offset, (void)count; return 0; }
int plat_mem_readable(uint64_t addr, uint64_t n) { (void)addr, (void)n; return 1; }
