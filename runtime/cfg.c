// totk.ini: file-backed defaults for the TOTK_* environment variables (docs/controls.md, totk.ini.example).
// Format: "key = value" lines, '#' or ';' at the start of a line begins a comment, [sections] only group. A key is a
// variable name without "TOTK_" (an optional TOTK_ prefix is accepted), case-insensitive. A value is the rest of the
// line, trimmed, optionally in double quotes; an empty value counts as unset. The environment wins over the file.
//
// Live settings: cfg_set changes a value in memory (readers see the new pointer; the old string is kept, so a pointer
// cfg_get handed out stays valid) and cfg_save writes only the changed keys back, keeping the rest of the file.
// Readers take no lock: an entry is complete before n_entries counts it, and value pointers are swapped atomically.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"

#define PREFIX "TOTK_"
#define PREFIX_LEN 5
#define CFG_MAX_KEYS 512

typedef struct { char *key, *value; int dirty, removed; } CfgEntry;   // key upper-case, without the prefix; value NULL after cfg_reset

static CfgEntry entries[CFG_MAX_KEYS];
static int n_entries;
static unsigned version;
static PlatMutex set_lock = PLAT_MUTEX_INIT;   // cfg_set, cfg_reset, cfg_save
static int save_ok = 1;                         // the last cfg_save_to wrote (or had nothing to write)

// Every name the runtime reads (and the patch sets in generated/), without the prefix. A file key outside this list is
// logged as unknown. A new variable goes here too, or its ini key logs a harmless "unknown key" line.
static const char *const known[] = {
    "AB", "APP_POOL", "ASPECT", "ASPECT_PATCH", "ASSUME_FOCUS", "ASTC", "ASTC_BC", "ASTC_TIME", "ASYNC_PREEMPT", "AUDIO", "AUDIO_CHANNELS", "AUDIO_OFF", "AUDIO_PREFILL_MS",
    "BACKGROUND_WINDOW", "BIND_A", "BIND_B", "BIND_DOWN", "BIND_L", "BIND_LCLICK", "BIND_LEFT", "BIND_LSTICK_DOWN", "BIND_LSTICK_LEFT",
    "BIND_LSTICK_RIGHT", "BIND_LSTICK_UP", "BIND_MINUS",
    "BIND_PLUS", "BIND_R", "BIND_RCLICK", "BIND_RIGHT", "BIND_RSTICK_DOWN", "BIND_RSTICK_LEFT", "BIND_RSTICK_RIGHT", "BIND_RSTICK_UP",
    "BIND_SKIP", "BIND_STATS", "BIND_UP", "BIND_X", "BIND_Y",
    "BIND_ZL", "BIND_ZR", "BLOCK_VIEWS_3D", "CLOTH_REALTIME", "COLOR_BRIGHTNESS", "COLOR_CONTRAST", "COLOR_GAMMA", "COLOR_PRESET", "COLOR_SATURATION", "CORE_DIAG", "CRIT", "CRUMB_BIND", "CRIT_OUT", "DESC_C0_SERIAL", "DESC_KEEP", "DESC_SAMP_GEN", "DESC_STAGE_ORIGIN", "DESC_TD_HASH", "DESC_TEMPLATE", "DEVICE_PROFILE", "DRAW_PROFILE", "DUMP", "DUMP_BUF", "DUMP_DIR", "DUMP_LOG", "DYN_UBO",
    "EXEFS", "EXPERIMENTAL_GEOMETRY", "FEEDBACK", "FIXED_STEP", "FIX_C0", "FIX_INDIRECT_RESET", "FIX_SCREEN_UV", "FIX_VS_FAULT", "FIX_ZERO", "FLIGHT", "FLIGHT_AT", "FLIGHT_DIR", "FLIGHT_FRAMES", "FLIGHT_SPIKE_MS",
    "FLIGHT_THUMBS", "FMZ", "FMZ_SEL", "FOV", "FOV_PATCH", "FPS60", "FPS_LIMIT", "FRAME_GEN", "FREEZE_IDLE", "FREE_CORES", "FULLSCREEN", "GPL", "GPL_DRAW_BUILD", "GPU",
    "GPU_INDEX", "GPU_NAME", "HDR", "HDR_GAME_NITS", "HDR_NATIVE", "HDR_PEAK_NITS", "HDR_SHOULDER", "HDR_TRACE", "HDR_UI_NITS", "HEAP_LAZY", "HIGHFPS_FIX", "HOST_PROPS_CACHE", "INPUT", "INPUT_RECORD", "INPUT_SCRIPT", "LANG", "LIB_PREBUILD",
    "LIB_PREBUILD_KEEP", "LIB_PREBUILD_THREADS", "LIST_RESTART", "LOCALE", "LOD_GUARD", "MEMORY", "MEM_REPORT", "MENU_LANG", "MENU_THEME", "MENU_TEST",
    "MENU_TEST_DIR", "MENU_TEST_SET", "MINIMAL_LAYOUT", "MIRROR_GAP", "MOTION", "MOUSE_CAMERA", "MOUSE_SENS", "MOVIE_HOST", "MOVIE_PACE", "MVK_UPSCALE_BLIT", "NAN_GUARD", "NATIVE_GS", "NX2_READ_NOWAIT",
    "NVN2_MODE_TRACE", "NVN2_MODE_TRACE_START_MS", "NVN2_MODE_TRACE_DURATION_MS", "ONE_PER_CORE", "OPERATION_MODE", "OUTPUT_RES", "PAD_LAYOUT", "PATCH_OUTPUT", "PATCH_SCENE", "PERF_OVERLAY", "PERF_SLOW_GPU_MS", "PERF_SLOW_MS", "PIPELINE_CACHE", "PIPE_MEMO", "PIPE_PREBUILD", "PIPE_SKIP", "PIPE_SYNC", "PIPE_WAIT", "POOL_FINI_WAIT", "POOL_LAZY",
    "PRESENT_HOLD", "PREWARM_XLATE", "PROBE_MOVIE", "PROBE_MOVIE_SHOT", "PROFILE", "PROFILE_CHAINS", "PROFILE_DIR", "PROFILE_FRAMES",
    "PROFILE_MAIN", "PROFILE_RENDER", "PROFILE_REPLAY", "PUSH_SKIP", "PUSH_STAGE", "RDOC", "RDOC_DIR", "RDOC_DLL", "REALTIME", "REC_BEGIN", "REC_END", "REC_SUBMIT", "REEXEC",
    "RENDER_SCALE", "REPLAY_ASYNC", "REPLAY_HELPERS", "REPLAY_PAR", "REPLAY_THREADS",
    "REPLAY_CHECK", "RESOLVE_CHECK", "ROMFS", "ROMFS_INDEX", "ROMFS_MMAP", "ROMFS_MODS", "ROMFS_PATCHES",
    "ROMFS_PROOF_DIR", "ROMFS_VERIFY", "RUMBLE", "RUMBLE_CURVE", "RUMBLE_STRENGTH", "RUMBLE_TRIGGERS", "SHADER_CACHE",
    "SHADER_MEMO", "SHARED_FONTS", "SHARPNESS", "SHOTS", "SHOTS_DIR", "SHOTS_FULL", "SMAA", "SPEED_PROBE", "SPIN", "SPIN_HANDOFF_US", "SPLIT_DRAWS", "SPLIT_US", "STICK_DEADZONE", "SVC_SLOW", "SWKBD_ANSWER",
    "SWKBD_TEST", "TEST_PAD", "TEST_PREPARE_SHADERS", "TEX_ALIAS", "TEX_HASH", "TEX_MEMO", "TEX_REPEAT", "TEX_VRAM_PRESSURE", "TIMER_1MS", "REC_BATCH", "TRACE", "UBO_SNAPSHOT", "UPSCALER", "VK_DYN_A2C", "VK_DYN_VERTEX", "VK_NO_FLOAT64",
    "VK_PRESENT", "VK_RECORDER", "VK_SSBO_ALIGN", "VK_SUBGROUP", "VK_SUBGROUP_PIN", "VK_VALIDATION", "VOLUME", "VRAM_HEADROOM", "VS_OUTS",
    "ULTRAWIDE", "HUD_POSITION", "WB_ASYNC", "WB_EXACT", "WB_FENCE_LIMIT", "WINDOW_MODE", "WINDOW_SIZE", "XLATE", "XLATE_ID", "XLATE_PROCS", "ZSTD_CHECK", "ZSTD_NATIVE",
};

// The totk.ini section a key cfg_set may write goes to when the file lacks it (totk.ini.example's sections).
static const struct { const char *key, *section; } sections[] = {
    {"FPS_LIMIT", "frame rate"}, {"FPS60", "frame rate"},
    {"REALTIME", "frame rate"}, {"CLOTH_REALTIME", "frame rate"}, {"HIGHFPS_FIX", "frame rate"},
    {"FULLSCREEN", "display"}, {"WINDOW_MODE", "display"}, {"WINDOW_SIZE", "display"}, {"PERF_OVERLAY", "display"}, {"VK_PRESENT", "display"},
    {"GPU_INDEX", "display"}, {"OUTPUT_RES", "display"}, {"RENDER_SCALE", "display"}, {"UPSCALER", "display"},
    {"ULTRAWIDE", "display"}, {"HUD_POSITION", "display"},
    {"SHARPNESS", "display"}, {"SMAA", "display"}, {"FRAME_GEN", "display"}, {"ASPECT", "display"}, {"FOV", "display"}, {"COLOR_PRESET", "display"},
    {"COLOR_BRIGHTNESS", "display"}, {"COLOR_CONTRAST", "display"}, {"COLOR_SATURATION", "display"},
    {"COLOR_GAMMA", "display"}, {"HDR", "display"}, {"HDR_PEAK_NITS", "display"}, {"HDR_GAME_NITS", "display"},
    {"HDR_UI_NITS", "display"}, {"HDR_NATIVE", "display"},
    {"DEVICE_PROFILE", "display"}, {"ASSUME_FOCUS", "display"},
    {"MENU_THEME", "menu"}, {"MENU_LANG", "language"}, {"LANG", "language"}, {"OPERATION_MODE", "language"},
    {"VOLUME", "audio"}, {"AUDIO_CHANNELS", "audio"},
    {"MOTION", "input"}, {"RUMBLE_STRENGTH", "input"}, {"RUMBLE_CURVE", "input"}, {"RUMBLE_TRIGGERS", "input"},
    {"PAD_LAYOUT", "input"}, {"STICK_DEADZONE", "input"}, {"MOUSE_CAMERA", "input"}, {"MOUSE_SENS", "input"}, {"INPUT_RECORD", "input"},
    {"BIND_A", "input"}, {"BIND_B", "input"}, {"BIND_X", "input"}, {"BIND_Y", "input"}, {"BIND_LCLICK", "input"},
    {"BIND_RCLICK", "input"}, {"BIND_L", "input"}, {"BIND_R", "input"}, {"BIND_ZL", "input"}, {"BIND_ZR", "input"},
    {"BIND_PLUS", "input"}, {"BIND_MINUS", "input"}, {"BIND_LEFT", "input"}, {"BIND_UP", "input"},
    {"BIND_RIGHT", "input"}, {"BIND_DOWN", "input"}, {"BIND_LSTICK_LEFT", "input"}, {"BIND_LSTICK_UP", "input"},
    {"BIND_LSTICK_RIGHT", "input"}, {"BIND_LSTICK_DOWN", "input"}, {"BIND_RSTICK_LEFT", "input"},
    {"BIND_RSTICK_UP", "input"}, {"BIND_RSTICK_RIGHT", "input"}, {"BIND_RSTICK_DOWN", "input"},
    {"ASYNC_PREEMPT", "performance"}, {"REPLAY_ASYNC", "performance"}, {"REPLAY_THREADS", "performance"}, {"FREE_CORES", "performance"},
    {"TIMER_1MS", "performance"}, {"PIPE_WAIT", "performance"}, {"PIPE_SKIP", "performance"},
    {"LIB_PREBUILD", "performance"}, {"LIB_PREBUILD_THREADS", "performance"}, {"PIPE_PREBUILD", "performance"},
    {"PREWARM_XLATE", "performance"}, {"GPL", "performance"}, {"ASTC", "performance"}, {"MEMORY", "performance"},
    {"EXPERIMENTAL_GEOMETRY", "performance"}, {"NATIVE_GS", "performance"},
    {"REC_BATCH", "performance"},
    {"TEX_ALIAS", "performance"}, {"TEX_VRAM_PRESSURE", "performance"}, {"ASTC_BC", "performance"}, {"LOD_GUARD", "performance"}, {"FIX_C0", "performance"}, {"FIX_INDIRECT_RESET", "performance"}, {"FIX_VS_FAULT", "performance"}, {"FIX_ZERO", "performance"}, {"FIX_SCREEN_UV", "performance"}, {"LIST_RESTART", "performance"},
    {"BIND_SKIP", "performance"}, {"SHADER_MEMO", "performance"}, {"PIPE_MEMO", "performance"}, {"TEX_MEMO", "performance"}, {"DESC_KEEP", "performance"}, {"DESC_TEMPLATE", "performance"},
    {"DESC_C0_SERIAL", "performance"}, {"DESC_SAMP_GEN", "performance"}, {"DESC_TD_HASH", "performance"}, {"DESC_STAGE_ORIGIN", "performance"}, {"TEX_REPEAT", "performance"},
    {"TEX_HASH", "performance"}, {"ZSTD_NATIVE", "performance"}, {"ZSTD_CHECK", "performance"}, {"MVK_UPSCALE_BLIT", "performance"},
    {"PUSH_SKIP", "performance"}, {"SPLIT_DRAWS", "performance"}, {"SPLIT_US", "performance"}, {"VK_VALIDATION", "performance"},
    {"VK_DYN_VERTEX", "performance"}, {"VK_DYN_A2C", "performance"}, {"VK_SUBGROUP_PIN", "performance"},
    {"TRACE", "diagnostics"}, {"MEM_REPORT", "diagnostics"}, {"HEAP_LAZY", "diagnostics"}, {"FLIGHT", "diagnostics"}, {"FLIGHT_SPIKE_MS", "diagnostics"},
    {"WB_FENCE_LIMIT", "diagnostics"}, {"WB_EXACT", "diagnostics"}, {"NX2_READ_NOWAIT", "diagnostics"},
    {"NVN2_MODE_TRACE", "diagnostics"}, {"HDR_TRACE", "diagnostics"}, {"HDR_SHOULDER", "diagnostics"},
    {"NVN2_MODE_TRACE_START_MS", "diagnostics"}, {"NVN2_MODE_TRACE_DURATION_MS", "diagnostics"},
};

// The settings the menu shows: the value in effect when the key is unset, and what the key accepts. The only place a
// default lives: consumers read cfg_value / cfg_int / cfg_double / cfg_bool. A row with an empty default has no fixed
// value (it depends on the machine); its consumer reads cfg_get and keeps an "unset" branch.
#define LIVE 1
#define RESTART 0
static const CfgMeta metas[] = {
    // Screen
    {"FULLSCREEN", "0", "0|1", LIVE}, {"WINDOW_MODE", "windowed", "windowed|borderless|fullscreen", LIVE},
    {"WINDOW_SIZE", "1920x1080", NULL, LIVE},
    {"VK_PRESENT", "mailbox", "mailbox|fifo|immediate", LIVE}, {"GPU_INDEX", "", NULL, RESTART},
    {"PERF_OVERLAY", "off", "off|fps|full", LIVE},
    {"OUTPUT_RES", "1920x1080", NULL, RESTART}, {"RENDER_SCALE", "game", "game|50|59|67|75|77|100|125|150|200", RESTART},
    {"UPSCALER", "fsr", "fsr|bilinear|dlss", LIVE}, {"SHARPNESS", "80", "0..100", LIVE}, {"SMAA", "off", "off|low|medium|high", LIVE},
    {"FRAME_GEN", "off", "off|2x", LIVE},   // vk_framegen.c, experimental
    {"ASPECT", "fit", "fit|stretch", LIVE}, {"FOV", "100", "70..140", LIVE},
    {"ULTRAWIDE", "0", "0|1", RESTART},
    {"HUD_POSITION", "edges", "edges|centered", RESTART},
    // colour adjustment (vk_color.c); the preset is the menu's, the four values are what the present uses
    {"COLOR_PRESET", "default", "default|vivid|clarity|custom", LIVE}, {"COLOR_BRIGHTNESS", "0", "-30..30", LIVE},
    {"COLOR_CONTRAST", "100", "70..150", LIVE}, {"COLOR_SATURATION", "100", "50..200", LIVE},
    {"COLOR_GAMMA", "1", "0.8..1.2", LIVE},
    // HDR output (vk_hdr.c, experimental): the swapchain format is chosen at start; the nits apply at once
    {"HDR", "0", "0|1", RESTART}, {"HDR_PEAK_NITS", "1000", "400..10000", LIVE}, {"HDR_GAME_NITS", "203", "80..500", LIVE},
    {"HDR_UI_NITS", "203", "80..500", LIVE},
    // Native HDR (ov_gfx.c, docs/hdr-native-plan.md) and its diagnostics: the call trace, and the switch of
    // the highlight rewrite (spirv_hdr_shoulder.c; 0 keeps the game's curve, about 1.36x paper white). Read at start.
    {"HDR_NATIVE", "0", "0|1", RESTART}, {"HDR_TRACE", "0", "0|1", RESTART},
    {"HDR_SHOULDER", "1", "0|1", RESTART},
    {"DEVICE_PROFILE", "auto", "auto|desktop|deck", RESTART},
    // Game
    {"FPS_LIMIT", "60", "30|40|45|60|90|120|144|display|unlimited", LIVE}, {"FPS60", "1", "0|1", LIVE},
    {"REALTIME", "1", "0|1", RESTART}, {"CLOTH_REALTIME", "1", "0|1", RESTART},
    {"HIGHFPS_FIX", "1", "0|1", RESTART},
    {"LANG", "", "ja|en-US|fr|de|it|es|zh-CN|ko|nl|pt|ru|zh-TW|en-GB|fr-CA|es-419|zh-Hans|zh-Hant|pt-BR", RESTART},
    {"OPERATION_MODE", "docked", "docked|handheld", RESTART},
    {"ASSUME_FOCUS", "1", "0|1", LIVE},   // read on every focus change (ov_system.c)
    // Sound, Controls, Menu
    {"VOLUME", "100", "0..100", LIVE}, {"AUDIO_CHANNELS", "auto", "auto|stereo|5.1", RESTART},
    {"MOTION", "1", "0|1", LIVE}, {"RUMBLE_STRENGTH", "130", "0..300", LIVE}, {"RUMBLE_CURVE", "0.6", "0.2..1.5", LIVE},
    {"RUMBLE_TRIGGERS", "0", "0|1", LIVE}, {"PAD_LAYOUT", "auto", "auto|nintendo|xbox|xbox_prompts", LIVE},
    {"STICK_DEADZONE", "8", "0..50", LIVE}, {"MOUSE_CAMERA", "0", "0|1", LIVE}, {"MOUSE_SENS", "1", "0..10", LIVE},
    {"MENU_LANG", "", "en|ru", LIVE}, {"MENU_THEME", "dk", "dk|classic", LIVE},
    // Advanced: scheduling
    {"ASYNC_PREEMPT", "1", "0|1|2", RESTART}, {"REPLAY_ASYNC", "0", "0|1", RESTART},
    {"REPLAY_THREADS", "0", "0..16", RESTART},   // sliced passes (vk_exec.c par_start): 0 off, else the recording threads
    {"FREE_CORES", "0", "0|1", RESTART}, {"TIMER_1MS", "1", "0|1", RESTART}, {"REC_BATCH", "0", "0..64", RESTART},
    // shaders and pipelines
    {"PIPE_WAIT", "default", "default|draw", RESTART}, {"PIPE_SKIP", "fullscreen", "strict|any|fullscreen", RESTART},
    {"LIB_PREBUILD", "1", "0|1", RESTART}, {"LIB_PREBUILD_THREADS", "", "1..64", RESTART},
    {"PIPE_PREBUILD", "1", "0|1", RESTART}, {"PREWARM_XLATE", "", "1..64", RESTART}, {"GPL", "", "0|1", RESTART},
    // textures and memory
    {"ASTC", "auto", "auto|gpu|native|cpu", RESTART}, {"MEMORY", "auto", "auto|import|mirror|verify", RESTART},
    {"TEX_ALIAS", "1", "0|1", RESTART}, {"TEX_HASH", "1", "0|1", RESTART}, {"TEX_VRAM_PRESSURE", "90", "0..100", RESTART}, {"ASTC_BC", "0", "0|1", RESTART}, {"ZSTD_NATIVE", "1", "0|1", RESTART}, {"ZSTD_CHECK", "0", "0|1", RESTART},
    // render fixes
    {"EXPERIMENTAL_GEOMETRY", "0", "0|1", RESTART}, {"NATIVE_GS", "1", "0|1", RESTART},
    {"MVK_UPSCALE_BLIT", "1", "0|1", RESTART},
    {"LOD_GUARD", "1", "0|1", RESTART}, {"FIX_C0", "1", "0|1", RESTART}, {"FIX_INDIRECT_RESET", "1", "0|1", RESTART}, {"FIX_VS_FAULT", "1", "0|1|all", RESTART}, {"FIX_ZERO", "1", "0|1", RESTART}, {"FIX_SCREEN_UV", "1", "0|1", RESTART}, {"LIST_RESTART", "1", "0|1", RESTART},
    {"BIND_SKIP", "1", "0|1", RESTART}, {"SHADER_MEMO", "1", "0|1", RESTART}, {"PIPE_MEMO", "1", "0|1", RESTART}, {"TEX_MEMO", "1", "0|1", RESTART},
    {"DESC_KEEP", "1", "0|1|ab", RESTART}, {"DESC_TEMPLATE", "1", "0|1", RESTART},
    {"DESC_C0_SERIAL", "0", "0|1|ab", RESTART}, {"DESC_SAMP_GEN", "0", "0|1|ab", RESTART}, {"DESC_TD_HASH", "0", "0|1|ab", RESTART},
    {"TEX_REPEAT", "1", "0|1|ab", RESTART},
    {"DESC_STAGE_ORIGIN", "0", "0|1|ab", RESTART},
    {"PUSH_SKIP", "1", "0|1", RESTART}, {"SPLIT_DRAWS", "64", "0..1000", RESTART},
    {"SPLIT_US", "1000", "0..20000", RESTART},
    // Vulkan
    {"VK_VALIDATION", "0", "0|1", RESTART}, {"VK_DYN_VERTEX", "", "0|1", RESTART},
    {"VK_DYN_A2C", "", "0|1", RESTART}, {"VK_SUBGROUP_PIN", "1", "0|1", RESTART},
    // diagnostics
    {"TRACE", "", NULL, RESTART}, {"MEM_REPORT", "0", "0..1000000", RESTART}, {"HEAP_LAZY", "0", "0|1", RESTART}, {"FLIGHT", "0", "0|1", RESTART}, {"FLIGHT_SPIKE_MS", "100", "0..60000", RESTART},
    {"WB_FENCE_LIMIT", "65536", "0..1073741824", RESTART}, {"WB_EXACT", "1", "0|1", RESTART},
    {"NX2_READ_NOWAIT", "0", "0|1", RESTART},
    {"NVN2_MODE_TRACE", "0", "0|1", RESTART},
    {"NVN2_MODE_TRACE_START_MS", "0", "0..600000", RESTART},
    {"NVN2_MODE_TRACE_DURATION_MS", "0", "0..600000", RESTART},
    {"INPUT_RECORD", "", NULL, RESTART},
};

static int is_known(const char *key) {
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (!strcmp(known[i], key)) return 1;
    return 0;
}

static void clear(void) {
    for (int i = 0; i < n_entries; i++) free(entries[i].key), free(entries[i].value);
    n_entries = 0;
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = 0;
    return s;
}

// The upper-case key without the prefix, in place.
static char *norm_key(char *key) {
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    return strncmp(key, PREFIX, PREFIX_LEN) ? key : key + PREFIX_LEN;
}

static void add(char *key, char *value, int line) {
    key = norm_key(key);
    size_t vl = strlen(value);
    if (vl >= 2 && value[0] == '"' && value[vl - 1] == '"') value[vl - 1] = 0, value++;
    if (!*key) return;
    if (!is_known(key)) rt_log("cfg: totk.ini line %d: unknown key '%s'", line, key);
    if (!*value) return;   // "key =" leaves the setting at its default
    for (int i = 0; i < n_entries; i++)
        if (!strcmp(entries[i].key, key)) {   // a repeated key: the later line wins
            free(entries[i].value);
            entries[i].value = strdup(value);
            return;
        }
    if (n_entries == CFG_MAX_KEYS) return;
    entries[n_entries++] = (CfgEntry){strdup(key), strdup(value), 0, 0};
}

int cfg_load(const char *path) {
    clear();
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data) return 0;
    char *p = (char *)data;
    if (size >= 3 && !memcmp(p, "\xEF\xBB\xBF", 3)) p += 3;   // editors add a BOM
    for (int line = 1; *p; line++) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *s = trim(p);
        char *eq = strchr(s, '=');
        if (*s && *s != '#' && *s != ';' && *s != '[') {
            if (!eq) rt_log("cfg: totk.ini line %d: no '=' in '%s'", line, s);
            else {
                *eq = 0;
                add(trim(s), trim(eq + 1), line);
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(data);
    return 1;
}

// ---- device profiles: defaults for a known machine, below the environment and totk.ini and above the table, so any
// key the player sets wins and resetting a key in the menu returns to the profile's value. TOTK_DEVICE_PROFILE = auto
// (default: the Steam Deck when the firmware says so), desktop (none) or deck.
//
// The Steam Deck (LCD Jupiter, OLED Galileo; 1280x800, 16:10, 4 Zen 2 cores, RDNA 2 with 8 CUs sharing 16 GB): the game
// at the screen's 1280x800 with its own 5/6 scene (the aspect patch gives the camera its 16:10, resolution.c, and the
// romfs-patches-1280x800 overlay the HUD's);
// fullscreen and vsync, as gamescope composes Game Mode; 40 fps, which divides 40 Hz (and 120 Hz on the OLED).
// Starting points to replace with measured ones (docs/tester-guide.md, Steam Deck).
static const struct { const char *key, *value; } DECK[] = {
    {"FULLSCREEN", "1"}, {"WINDOW_SIZE", "1280x800"}, {"OUTPUT_RES", "1280x800"}, {"RENDER_SCALE", "game"},
    {"VK_PRESENT", "fifo"}, {"FPS_LIMIT", "40"},
};
static int profile = -1;   // 0 none, 1 Steam Deck

static int read_line(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(out, (int)n, f) != NULL;
    fclose(f);
    out[ok ? strcspn(out, "\r\n") : 0] = 0;
    return ok;
}

// The Steam Deck: Valve's board, product Jupiter or Galileo (/sys/class/dmi/id); SteamDeck=1, which Steam sets on it.
static int on_steam_deck(void) {
    const char *env = getenv("SteamDeck");
    if (env && !strcmp(env, "1")) return 1;
    char vendor[64], product[64];
    return read_line("/sys/class/dmi/id/board_vendor", vendor, sizeof vendor) && !strcmp(vendor, "Valve") &&
           read_line("/sys/class/dmi/id/product_name", product, sizeof product) &&
           (!strcmp(product, "Jupiter") || !strcmp(product, "Galileo"));
}

static const char *raw_get(const char *name);

static int profile_of(void) {
    int p = __atomic_load_n(&profile, __ATOMIC_ACQUIRE);
    if (p >= 0) return p;
    const char *set = raw_get(PREFIX "DEVICE_PROFILE");
    if (set && !strcmp(set, "deck")) p = 1;
    else if (set && *set && strcmp(set, "auto")) p = 0;
    else p = on_steam_deck();
    __atomic_store_n(&profile, p, __ATOMIC_RELEASE);
    return p;
}

const char *cfg_device_profile(void) { return profile_of() ? "deck" : "desktop"; }

// DKB_X in the environment stands for TOTK_X and wins over it: the TOTK_ names come from TotK Recomp and stay, so its
// fixes keep applying without a rename.
static const char *dkb_alias(const char *name) {
    char alias[96];
    if (strncmp(name, PREFIX, PREFIX_LEN) || strlen(name) >= sizeof alias - 1) return NULL;
    snprintf(alias, sizeof alias, "DKB_%s", name + PREFIX_LEN);
    return getenv(alias);
}

static const char *env_of(const char *name) {
    const char *e = dkb_alias(name);
    return e ? e : getenv(name);
}

static const char *raw_get(const char *name) {
    const char *e = env_of(name);
    if (e || strncmp(name, PREFIX, PREFIX_LEN)) return e;
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (!strcasecmp(entries[i].key, name + PREFIX_LEN)) return __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
    return NULL;
}

const char *cfg_get(const char *name) {
    const char *v = raw_get(name);
    if (v || strncmp(name, PREFIX, PREFIX_LEN) || !profile_of()) return v;
    for (size_t i = 0; i < sizeof DECK / sizeof *DECK; i++)
        if (!strcasecmp(DECK[i].key, name + PREFIX_LEN)) return DECK[i].value;
    return NULL;
}

// Whether name's value comes from the player (the environment or totk.ini), not a profile or the table.
int cfg_is_set(const char *name) { return raw_get(name) != NULL; }

unsigned cfg_version(void) { return __atomic_load_n(&version, __ATOMIC_ACQUIRE); }

int cfg_from_env(const char *name) { return env_of(name) != NULL; }

void cfg_set(const char *name, const char *value) {
    if (strncmp(name, PREFIX, PREFIX_LEN) || !value) return;
    char key[64];
    snprintf(key, sizeof key, "%s", name + PREFIX_LEN);
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    plat_mutex_lock(&set_lock);
    int i = 0, n = n_entries;
    while (i < n && strcmp(entries[i].key, key)) i++;
    if (i < n) {
        if (!entries[i].value || strcmp(entries[i].value, value)) {
            __atomic_store_n(&entries[i].value, strdup(value), __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 1;
        }
        entries[i].removed = 0;   // a value set after cfg_reset replaces the line instead of deleting it
    } else if (n < CFG_MAX_KEYS) {
        entries[n] = (CfgEntry){strdup(key), strdup(value), 1, 0};
        __atomic_store_n(&n_entries, n + 1, __ATOMIC_RELEASE);
    }
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
}

int cfg_reset(const char *const *keys, int n) {
    int done = 0;
    plat_mutex_lock(&set_lock);
    for (int k = 0; k < n; k++) {
        char key[64], var[80];
        snprintf(key, sizeof key, "%s", keys[k]);
        const char *name = norm_key(key);
        snprintf(var, sizeof var, PREFIX "%s", name);
        if (env_of(var)) continue;   // the environment wins; the file line stays
        int i = 0;
        while (i < n_entries && strcmp(entries[i].key, name)) i++;
        if (i == n_entries) {   // no value in memory ("key =" in the file): the line still has to go
            if (i == CFG_MAX_KEYS) continue;
            entries[i] = (CfgEntry){strdup(name), NULL, 0, 1};
            __atomic_store_n(&n_entries, i + 1, __ATOMIC_RELEASE);
        } else if (entries[i].value) {
            __atomic_store_n(&entries[i].value, NULL, __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 0;
            entries[i].removed = 1;   // cfg_save_to deletes the key's lines
            done++;
        } else {
            entries[i].removed = 1;
        }
    }
    if (done) __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
    return done;
}

int cfg_last_save_ok(void) { return __atomic_load_n(&save_ok, __ATOMIC_ACQUIRE); }

void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx) {
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        const char *v = __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
        if (v) cb(entries[i].key, v, 0, ctx);
    }
    for (size_t i = 0; i < sizeof known / sizeof *known; i++) {
        char var[80];
        snprintf(var, sizeof var, PREFIX "%s", known[i]);
        const char *e = env_of(var);
        if (e) cb(known[i], e, 1, ctx);
    }
}

const CfgMeta *cfg_meta_at(int i) {
    if (i < 0 || (size_t)i >= sizeof metas / sizeof *metas) return NULL;
    // Bananza's virtual display is 2880x1620; keep its stock resolution when
    // no setting is present, including the menu's default marker and Reset.
    static const CfgMeta dkb_output = {"OUTPUT_RES", "2880x1620", NULL, RESTART};
    if (!rt_game_version && !strcmp(metas[i].key, "OUTPUT_RES")) return &dkb_output;
    // Bananza timing is experimental; TotK retains its established default.
    static const CfgMeta dkb_realtime = {"REALTIME", "0", "0|1", RESTART};
    if (!rt_game_version && !strcmp(metas[i].key, "REALTIME")) return &dkb_realtime;
    return &metas[i];
}

const CfgMeta *cfg_meta(const char *key) {
    char k[64];
    snprintf(k, sizeof k, "%s", key);
    const char *name = norm_key(k);
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++)
        if (!strcmp(metas[i].key, name)) return cfg_meta_at((int)i);
    return NULL;
}

const char *cfg_value(const char *name) {
    const char *e = cfg_get(name);
    if (e && *e) return e;
    const CfgMeta *m = cfg_meta(name);
    if (!m) rt_fatal("cfg: %s has no row in the settings table", name);
    return m->def;
}

// The row's default as a number, for a set value that does not parse.
static double default_number(const char *name) {
    return strtod(cfg_meta(name)->def, NULL);
}

double cfg_double(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    double d = strtod(v, &end);
    return end == v || *end || d != d ? default_number(name) : d;
}

int cfg_int(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    long l = strtol(v, &end, 10);
    return end == v || *end ? (int)default_number(name) : (int)l;
}

int cfg_bool(const char *name) { return strcmp(cfg_value(name), "0") != 0; }

// ---- cfg_save: the file as lines, each with its line ending
typedef struct { const char *s; size_t n; } Line;   // n includes the '\n' (and '\r') when present

static int line_count(const char *p, size_t size) {
    int n = 0;
    for (size_t i = 0; i < size; i++) n += p[i] == '\n';
    return n + 1;
}

// The line's text without its ending, BOM and surrounding blanks, into out.
static void line_text(const Line *l, int first, char *out, size_t cap) {
    size_t n = l->n;
    const char *s = l->s;
    if (first && n >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) s += 3, n -= 3;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
    char *t = trim(out);
    memmove(out, t, strlen(t) + 1);
}

// The key of a "key = value" line (comment = 0) or of a commented "# key = value" line (comment = 1), normalised,
// into key; 0 when the line is not one.
static int line_key(const char *text, int comment, char *key, size_t cap) {
    if (comment) {
        if (*text != '#' && *text != ';') return 0;
        text++;
        while (*text == ' ' || *text == '\t') text++;
    }
    if (!*text || *text == '#' || *text == ';' || *text == '[') return 0;
    const char *eq = strchr(text, '=');
    if (!eq) return 0;
    size_t n = (size_t)(eq - text);
    while (n && (text[n - 1] == ' ' || text[n - 1] == '\t')) n--;
    if (!n || n >= cap) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)text[i]) && text[i] != '_') return 0;
    char tmp[128];
    memcpy(tmp, text, n);
    tmp[n] = 0;
    snprintf(key, cap, "%s", norm_key(tmp));
    return 1;
}

// The section name of a "[name]" line, lower-case, into out; 0 when the line is not one.
static int line_section(const char *text, char *out, size_t cap) {
    if (*text != '[') return 0;
    const char *e = strchr(text, ']');
    if (!e) return 0;
    char tmp[128];
    size_t n = (size_t)(e - text - 1);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, text + 1, n);
    tmp[n] = 0;
    snprintf(out, cap, "%s", trim(tmp));
    for (char *p = out; *p; p++) *p = (char)tolower((unsigned char)*p);
    return 1;
}

typedef struct { char *buf; size_t n, cap; } Out;
static void put(Out *o, const char *s, size_t n) {
    if (o->n + n + 1 > o->cap) {
        o->cap = (o->n + n + 1) * 2;
        o->buf = realloc(o->buf, o->cap);
        if (!o->buf) rt_fatal("cfg: out of memory writing totk.ini");
    }
    memcpy(o->buf + o->n, s, n);
    o->n += n;
}
static void puts_(Out *o, const char *s) { put(o, s, strlen(s)); }

static const char *section_of(const char *key) {
    for (size_t i = 0; i < sizeof sections / sizeof *sections; i++)
        if (!strcmp(sections[i].key, key)) return sections[i].section;
    return NULL;
}

// "key = value" + eol for entry e (key lower-case)
static void put_entry(Out *o, const CfgEntry *e, const char *eol) {
    char k[64];
    snprintf(k, sizeof k, "%s", e->key);
    for (char *p = k; *p; p++) *p = (char)tolower((unsigned char)*p);
    puts_(o, k);
    puts_(o, " = ");
    puts_(o, e->value);
    puts_(o, eol);
}

int cfg_save_to(const char *path) {
    plat_mutex_lock(&set_lock);
    int n = n_entries, any = 0;
    for (int i = 0; i < n; i++) any |= entries[i].dirty | entries[i].removed;
    if (!any) {
        __atomic_store_n(&save_ok, 1, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 1;
    }
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data && plat_file_exists(path) && plat_file_size(path) != 0) {
        // the file is there but unreadable (locked, transient error): rebuilding it from nothing would lose its lines
        rt_log("cfg: cannot read %s, nothing written", path);
        __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 0;
    }
    const char *text = data ? (const char *)data : "";
    int nl = data ? line_count(text, size) : 0;
    Line *lines = calloc((size_t)nl + 1, sizeof *lines);
    // after[i]: entries to write after line i (-1 = none); replace[i]: the entry whose value line i now carries
    int *replace = malloc(sizeof(int) * ((size_t)nl + 1)), *after_key = malloc(sizeof(int) * (size_t)(n + 1));
    char *drop = calloc((size_t)nl + 1, 1);   // lines of a reset key
    if (!lines || !replace || !after_key || !drop) rt_fatal("cfg: out of memory writing totk.ini");
    for (int i = 0, at = 0; i < nl; i++) {
        size_t start = (size_t)at;
        while ((uint64_t)at < size && text[at] != '\n') at++;
        if ((uint64_t)at < size) at++;
        lines[i] = (Line){text + start, (size_t)at - start};
        replace[i] = -1;
    }
    if (nl && !lines[nl - 1].n) nl--;   // the text ended with a newline: no last empty line
    const char *eol = data && strstr(text, "\r\n") ? "\r\n" : "\n";
    char t[1024], k[128], sec[128];
    for (int e = 0; e < n; e++) {
        after_key[e] = -2;   // -2 clean, -1 append in a new section at the end, >= 0 insert after that line
        if (entries[e].removed)
            for (int i = 0; i < nl; i++) {
                line_text(&lines[i], i == 0, t, sizeof t);
                if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) drop[i] = 1;
            }
        if (!entries[e].dirty) continue;
        int last = -1;
        for (int i = 0; i < nl; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) last = i;
        }
        if (last >= 0) {
            replace[last] = e;
            continue;
        }
        const char *want = section_of(entries[e].key);
        int in = 0, example = -1, tail = -1;
        for (int i = 0; i < nl && want; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_section(t, sec, sizeof sec)) {
                in = !strcmp(sec, want);
                if (in) tail = i;
                continue;
            }
            if (!in) continue;
            if (*t) tail = i;
            if (line_key(t, 1, k, sizeof k) && !strcmp(k, entries[e].key)) example = i;
        }
        after_key[e] = example >= 0 ? example : tail;
    }
    Out o = {0};
    for (int i = 0; i < nl; i++) {
        if (drop[i]) {
            // the line leaves the file; keys inserted after it still need a line ending before them
        } else if (replace[i] >= 0) {
            // keep the line's indentation and key spelling, put the new value after " = "
            const char *s = lines[i].s;
            size_t len = lines[i].n;
            if (i == 0 && len >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) put(&o, s, 3), s += 3, len -= 3;
            const char *eq = memchr(s, '=', len);
            size_t kn = (size_t)(eq - s);
            while (kn && (s[kn - 1] == ' ' || s[kn - 1] == '\t')) kn--;
            put(&o, s, kn);
            puts_(&o, " = ");
            puts_(&o, entries[replace[i]].value);
            int had_eol = len && s[len - 1] == '\n';
            if (had_eol) puts_(&o, eol);
        } else {
            put(&o, lines[i].s, lines[i].n);
        }
        for (int e = 0; e < n; e++)
            if (after_key[e] == i) {
                if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
                put_entry(&o, &entries[e], eol);
            }
    }
    // keys whose section the file lacks: one new section each, at the end
    for (int e = 0; e < n; e++) {
        if (after_key[e] != -1) continue;
        const char *want = section_of(entries[e].key);
        if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
        if (want) {
            if (o.n) puts_(&o, eol);
            puts_(&o, "[");
            puts_(&o, want);
            puts_(&o, "]");
            puts_(&o, eol);
        }
        put_entry(&o, &entries[e], eol);
        // the following keys of the same new section go right after this one
        for (int f = e + 1; f < n && want; f++)
            if (after_key[f] == -1 && section_of(entries[f].key) && !strcmp(section_of(entries[f].key), want)) {
                put_entry(&o, &entries[f], eol);
                after_key[f] = -3;
            }
    }
    int ok = plat_file_write_atomic(path, o.buf ? o.buf : "", o.n);
    if (ok)
        for (int e = 0; e < n; e++) entries[e].dirty = entries[e].removed = 0;
    else
        rt_log("cfg: cannot write %s", path);
    __atomic_store_n(&save_ok, ok ? 1 : 0, __ATOMIC_RELEASE);
    free(o.buf);
    free(drop);
    free(after_key);
    free(replace);
    free(lines);
    free(data);
    plat_mutex_unlock(&set_lock);
    return ok;
}

static char ini_path[1024];

static int find_ini(void) {
    if (ini_path[0]) return 1;
    char path[1024];
    if (!plat_exe_path(path, sizeof path)) return 0;
    char *cut = plat_last_sep(path);
    if (!cut) cut = strrchr(path, '/');
    if (!cut || cut + 10 >= path + sizeof path) return 0;
    // Bananza writes dkb.ini and reads an older totk.ini only while no dkb.ini exists; TotK keeps totk.ini.
    strcpy(cut + 1, rt_game_version ? "totk.ini" : "dkb.ini");
    snprintf(ini_path, sizeof ini_path, "%s", path);
    if (!rt_game_version && !plat_file_exists(ini_path)) {
        // No dkb.ini yet: an existing totk.ini is copied to it, so the settings carry over and every later write
        // (which only touches the changed lines) lands in dkb.ini with the rest of the file intact.
        strcpy(cut + 1, "totk.ini");
        uint64_t size = 0;
        uint8_t *old = plat_file_exists(path) ? plat_file_read(path, &size) : NULL;
        if (old) {
            if (!plat_file_write_atomic(ini_path, old, size)) {
                // The copy failed: keep using totk.ini, so the old settings still load and later saves update it.
                rt_log("cfg: cannot copy %s to dkb.ini, keeping totk.ini", path);
                snprintf(ini_path, sizeof ini_path, "%s", path);
            }
        }
        free(old);
    }
    return 1;
}

int cfg_save(void) {
    if (find_ini()) return cfg_save_to(ini_path);
    __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
    return 0;
}

static void load_ini(void) {
    if (!find_ini()) return;
    if (!cfg_load(ini_path)) {
        rt_log("cfg: no settings file at %s", ini_path);
        return;
    }
    rt_log("cfg: read %s: %d keys", ini_path, n_entries);
    for (int i = 0; i < n_entries; i++) {
        char var[128];
        snprintf(var, sizeof var, PREFIX "%s", entries[i].key);
        if (env_of(var)) rt_log("cfg: %s (or its DKB_ name) is set in the environment and overrides the settings file", var);
    }
}

void cfg_init(void) {
    load_ini();
    if (profile_of())
        rt_log("cfg: device profile Steam Deck: defaults for the keys the settings file does not set (TOTK_DEVICE_PROFILE=desktop "
               "turns it off)");
}
