// Ported from adriadam10/mkw-recomp (MIT, Copyright (c) 2026 Adrià), runtime/input/linux_switch2.c at eb8fe198; TOTK_SWITCH2 for MKW_SWITCH2.
// Switch 2 controllers over Bluetooth LE on Linux (switch2.h): BlueZ over the system D-Bus, with libdbus-1 loaded at run
// time as SDL loads it (no headers: the few calls and the two structs used are declared below, after dbus 1.12's
// dbus-message.h and dbus-errors.h). One thread: LE discovery, Nintendo's advertisements (company id 0x0553), connect,
// the GATT characteristics by their UUIDs, notifications in, commands out as writes without response.
// Without BlueZ, libdbus or an adapter it logs why and stops; the rest of the input is untouched.
#if defined(__linux__)   // build_game.py compiles every runtime/*.c; this transport is for Linux only
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "switch2.h"
#include "switch2_proto.h"

typedef struct DBusConnection DBusConnection;
typedef struct DBusMessage DBusMessage;
typedef struct { void *dummy1, *dummy2; uint32_t dummy3; int dummy4[8], pad1; void *pad2, *pad3; } DBusMessageIter;
typedef struct { const char *name, *message; unsigned dummy; void *padding1; } DBusError;
enum { BUS_SYSTEM = 1, T_INVALID = 0, T_ARRAY = 'a', T_DICT = 'e', T_VARIANT = 'v', T_STRING = 's', T_PATH = 'o',
       T_BOOL = 'b', T_BYTE = 'y', T_UINT16 = 'q' };

static struct {
    DBusConnection *(*bus_get_private)(int, DBusError *);
    void (*error_init)(DBusError *);
    unsigned (*error_is_set)(const DBusError *);
    void (*error_free)(DBusError *);
    DBusMessage *(*new_method_call)(const char *, const char *, const char *, const char *);
    DBusMessage *(*send_block)(DBusConnection *, DBusMessage *, int, DBusError *);
    unsigned (*send)(DBusConnection *, DBusMessage *, uint32_t *);
    void (*unref)(DBusMessage *);
    void (*add_match)(DBusConnection *, const char *, DBusError *);
    unsigned (*read_write)(DBusConnection *, int);
    DBusMessage *(*pop)(DBusConnection *);
    unsigned (*is_signal)(DBusMessage *, const char *, const char *);
    const char *(*get_path)(DBusMessage *);
    void (*init_append)(DBusMessage *, DBusMessageIter *);
    unsigned (*append_basic)(DBusMessageIter *, int, const void *);
    unsigned (*open)(DBusMessageIter *, int, const char *, DBusMessageIter *);
    unsigned (*close)(DBusMessageIter *, DBusMessageIter *);
    unsigned (*append_fixed)(DBusMessageIter *, int, const void *, int);
    unsigned (*iter_init)(DBusMessage *, DBusMessageIter *);
    int (*arg_type)(DBusMessageIter *);
    void (*get_basic)(DBusMessageIter *, void *);
    void (*recurse)(DBusMessageIter *, DBusMessageIter *);
    unsigned (*next)(DBusMessageIter *);
    void (*get_fixed)(DBusMessageIter *, void *, int *);
    void (*flush)(DBusConnection *);
} D;

static int load_dbus(void) {
    void *h = dlopen("libdbus-1.so.3", RTLD_NOW | RTLD_LOCAL);
    if (!h) return 0;
    static const char *const names[] = {
        "dbus_bus_get_private", "dbus_error_init", "dbus_error_is_set", "dbus_error_free",
        "dbus_message_new_method_call", "dbus_connection_send_with_reply_and_block", "dbus_connection_send",
        "dbus_message_unref", "dbus_bus_add_match", "dbus_connection_read_write", "dbus_connection_pop_message",
        "dbus_message_is_signal", "dbus_message_get_path", "dbus_message_iter_init_append",
        "dbus_message_iter_append_basic", "dbus_message_iter_open_container", "dbus_message_iter_close_container",
        "dbus_message_iter_append_fixed_array", "dbus_message_iter_init", "dbus_message_iter_get_arg_type",
        "dbus_message_iter_get_basic", "dbus_message_iter_recurse", "dbus_message_iter_next",
        "dbus_message_iter_get_fixed_array", "dbus_connection_flush",
    };
    void **slot = (void **)&D;
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (!(slot[i] = dlsym(h, names[i]))) return 0;
    return 1;
}

static DBusConnection *bus;
static char adapter[128];
static uint8_t host[6];   // the adapter's address, little endian (what the controllers wake toward)

struct S2Link {
    char dev[160], cmd[200], rsp[200], input[200], vib[200], rate[240];   // rate: the report-rate descriptor, if any
    uint16_t pid;
    int use;
    int gone;   // disconnected while its handshake ran: dropped once that returns
    S2Session *s;
    struct S2Link *next;
};
static S2Link *busy;   // the link whose handshake is running (its replies come through pump)
static int trace;      // TOTK_SWITCH2=2
static S2Link *links;

// Devices seen advertising for us and not yet tried, and those that failed (not tried again for 10 s)
enum { MAX_PENDING = 8 };
static struct { char path[160]; uint16_t pid; int use; } pending[MAX_PENDING];
static int npending;
static struct { char path[160]; uint64_t until; } failed[MAX_PENDING];

// ---- calls

static DBusMessage *call(DBusMessage *m, int ms) {
    DBusError e;
    D.error_init(&e);
    DBusMessage *r = D.send_block(bus, m, ms, &e);
    D.unref(m);
    if (D.error_is_set(&e)) {
        rt_log("input: Switch 2 (BlueZ): %s", e.message ? e.message : e.name);
        D.error_free(&e);
    }
    return r;
}

static DBusMessage *method(const char *path, const char *iface, const char *name) {
    return D.new_method_call("org.bluez", path, iface, name);
}

static int call_void(const char *path, const char *iface, const char *name, int ms) {
    DBusMessage *r = call(method(path, iface, name), ms);
    if (r) D.unref(r);
    return r != NULL;
}

// a{sv} entry with a string or boolean value
static void dict_entry(DBusMessageIter *a, const char *key, int type, const void *v) {
    DBusMessageIter e, var;
    char sig[2] = {(char)type, 0};
    D.open(a, T_DICT, NULL, &e);
    D.append_basic(&e, T_STRING, &key);
    D.open(&e, T_VARIANT, sig, &var);
    D.append_basic(&var, type, v);
    D.close(&e, &var);
    D.close(a, &e);
}

static int write_value(const char *path, const uint8_t *d, size_t n) {
    DBusMessage *m = method(path, "org.bluez.GattCharacteristic1", "WriteValue");
    DBusMessageIter it, arr, opts;
    D.init_append(m, &it);
    D.open(&it, T_ARRAY, "y", &arr);
    D.append_fixed(&arr, T_BYTE, &d, (int)n);
    D.close(&it, &arr);
    D.open(&it, T_ARRAY, "{sv}", &opts);
    const char *type = "command";   // write without response
    dict_entry(&opts, "type", T_STRING, &type);
    D.close(&it, &opts);
    int ok = D.send(bus, m, NULL) != 0;
    D.unref(m);
    return ok;
}

static int link_write_cmd(S2Link *l, const uint8_t *d, size_t n) { return write_value(l->cmd, d, n); }
static int link_write_vib(S2Link *l, const uint8_t *d, size_t n) { return l->vib[0] && write_value(l->vib, d, n); }
// The report-rate descriptor of the input characteristic written, when the controller has one (switch2_proto.h)
static void write_rate_desc(S2Link *l) {
    static const uint8_t pay[] = {S2_RATE_PAYLOAD};
    size_t n = strlen(l->input);
    const char *what = "not found";
    if (l->rate[0] && !strncmp(l->rate, l->input, n) && l->rate[n] == '/') {
        DBusMessage *m = method(l->rate, "org.bluez.GattDescriptor1", "WriteValue");
        DBusMessageIter it, arr, opts;
        const uint8_t *p = pay;
        D.init_append(m, &it);
        D.open(&it, T_ARRAY, "y", &arr);
        D.append_fixed(&arr, T_BYTE, &p, (int)sizeof pay);
        D.close(&it, &arr);
        D.open(&it, T_ARRAY, "{sv}", &opts);
        D.close(&it, &opts);
        DBusMessage *r = call(m, 5000);
        what = r ? "written" : "refused";
        if (r) D.unref(r);
    }
    rt_log("input: Switch 2 controller %s: report-rate descriptor %s", l->dev, what);
}

static int link_start_input(S2Link *l) {
    write_rate_desc(l);
    return call_void(l->input, "org.bluez.GattCharacteristic1", "StartNotify", 5000);
}
static void pump(int ms);
static const S2LinkOps ops = {link_write_cmd, link_write_vib, link_start_input, pump};

// ---- reading replies and signals

// The variant at *v: its first value's iterator in *in
static int variant(DBusMessageIter *v, DBusMessageIter *in) {
    if (D.arg_type(v) != T_VARIANT) return 0;
    D.recurse(v, in);
    return 1;
}

// A byte array (ay) at it
static int bytes(DBusMessageIter *it, const uint8_t **d, int *n) {
    DBusMessageIter a;
    if (D.arg_type(it) != T_ARRAY) return 0;
    D.recurse(it, &a);
    *d = NULL, *n = 0;
    if (D.arg_type(&a) == T_BYTE) D.get_fixed(&a, d, n);
    return 1;
}

// "AA:BB:CC:DD:EE:FF" to six bytes, little endian (the last pair first)
static int parse_address(const char *s, uint8_t out[6]) {
    unsigned b[6];
    if (sscanf(s, "%2x:%2x:%2x:%2x:%2x:%2x", &b[5], &b[4], &b[3], &b[2], &b[1], &b[0]) != 6) return 0;
    for (int i = 0; i < 6; i++) out[i] = (uint8_t)b[i];
    return 1;
}

static S2Link *link_by_char(const char *path, int *is_input) {
    for (S2Link *l = links; l; l = l->next) {
        if (!strcmp(path, l->input)) return *is_input = 1, l;
        if (!strcmp(path, l->rsp)) return *is_input = 0, l;
    }
    return NULL;
}

static S2Link *link_by_dev(const char *path) {
    for (S2Link *l = links; l; l = l->next)
        if (!strcmp(path, l->dev)) return l;
    return NULL;
}

static void drop(S2Link *l) {
    S2Link **p = &links;
    while (*p && *p != l) p = &(*p)->next;
    if (*p) *p = l->next;
    if (l->s) switch2_close(l->s);
    free(l);
}

// TOTK_SWITCH2=2 logs each device once: BlueZ repeats an advertisement every time it hears it (DuplicateData)
static int first_seen(const char *path) {
    static char seen[64][160];
    static int n;
    for (int i = 0; i < n; i++)
        if (!strcmp(seen[i], path)) return 0;
    if (n < 64) snprintf(seen[n++], sizeof seen[0], "%s", path);
    return 1;
}

// Device1 properties (a{sv}) of path: Nintendo's manufacturer data queues it for a connection
static void device_props(const char *path, DBusMessageIter *props) {
    for (; D.arg_type(props) == T_DICT; D.next(props)) {
        DBusMessageIter e, v, m;
        const char *key;
        D.recurse(props, &e);
        D.get_basic(&e, &key);
        D.next(&e);
        if (!strcmp(key, "Connected")) {
            unsigned on = 1;
            if (variant(&e, &v) && D.arg_type(&v) == T_BOOL) D.get_basic(&v, &on);
            S2Link *l = on ? NULL : link_by_dev(path);
            if (l && l == busy) l->gone = 1;
            else if (l) drop(l);
            continue;
        }
        if (strcmp(key, "ManufacturerData") || !variant(&e, &v) || D.arg_type(&v) != T_ARRAY) continue;
        D.recurse(&v, &m);
        for (; D.arg_type(&m) == T_DICT; D.next(&m)) {
            DBusMessageIter me, mv;
            uint16_t company;
            const uint8_t *d;
            int n;
            s2_advert a;
            D.recurse(&m, &me);
            D.get_basic(&me, &company);
            D.next(&me);
            if (!variant(&me, &mv) || !bytes(&mv, &d, &n)) continue;
            if (trace && first_seen(path))   // TOTK_SWITCH2=2: each device with manufacturer data once, for a report
                rt_log("input: Switch 2 trace: %s company 0x%04x, %d bytes%s", path, company, n,
                       company == S2_COMPANY_ID ? (s2_advert_parse(d, (size_t)n, &a) ? ", a Switch 2 controller" : ", Nintendo, not a Switch 2 controller") : "");
            if (company != S2_COMPANY_ID || !s2_advert_parse(d, (size_t)n, &a)) continue;
            int use = s2_advert_use(&a, host);
            if (use == S2_ADV_IGNORE || link_by_dev(path)) continue;
            int dup = 0;
            for (int i = 0; i < npending; i++) dup |= !strcmp(pending[i].path, path);
            for (int i = 0; i < MAX_PENDING; i++) dup |= !strcmp(failed[i].path, path) && switch2_now_ns() < failed[i].until;
            if (dup || npending == MAX_PENDING) continue;
            snprintf(pending[npending].path, sizeof pending[npending].path, "%s", path);
            pending[npending].pid = a.pid, pending[npending].use = use;
            npending++;
        }
    }
}

static void on_signal(DBusMessage *m) {
    DBusMessageIter it, sub;
    const char *iface, *path = D.get_path(m);
    if (!path || !D.iter_init(m, &it)) return;
    if (D.is_signal(m, "org.freedesktop.DBus.Properties", "PropertiesChanged")) {
        D.get_basic(&it, &iface);
        D.next(&it);
        if (D.arg_type(&it) != T_ARRAY) return;
        D.recurse(&it, &sub);
        if (!strcmp(iface, "org.bluez.Device1")) return device_props(path, &sub);
        int is_input;
        S2Link *l = strcmp(iface, "org.bluez.GattCharacteristic1") ? NULL : link_by_char(path, &is_input);
        for (; l && D.arg_type(&sub) == T_DICT; D.next(&sub)) {
            DBusMessageIter e, v;
            const char *key;
            const uint8_t *d;
            int n;
            D.recurse(&sub, &e);
            D.get_basic(&e, &key);
            D.next(&e);
            if (strcmp(key, "Value") || !variant(&e, &v) || !bytes(&v, &d, &n)) continue;
            if (!l->s) continue;
            if (is_input) switch2_report(l->s, d, (size_t)n, switch2_now_ns());
            else switch2_reply(l->s, d, (size_t)n);
        }
    } else if (D.is_signal(m, "org.freedesktop.DBus.ObjectManager", "InterfacesAdded")) {
        const char *obj;
        D.get_basic(&it, &obj);
        D.next(&it);
        if (D.arg_type(&it) != T_ARRAY) return;
        D.recurse(&it, &sub);
        for (; D.arg_type(&sub) == T_DICT; D.next(&sub)) {
            DBusMessageIter e, props;
            D.recurse(&sub, &e);
            D.get_basic(&e, &iface);
            D.next(&e);
            if (!strcmp(iface, "org.bluez.Device1") && D.arg_type(&e) == T_ARRAY) {
                D.recurse(&e, &props);
                device_props(obj, &props);
            }
        }
    }
}

static void pump(int ms) {
    D.read_write(bus, ms);
    for (DBusMessage *m; (m = D.pop(bus)); D.unref(m)) on_signal(m);
}

// ---- objects: BlueZ's GetManagedObjects (a{oa{sa{sv}}}), each interface's properties handed to fn

typedef void (*ObjectFn)(const char *path, const char *iface, DBusMessageIter *props, void *ctx);

static int managed(ObjectFn fn, void *ctx) {
    DBusMessage *r = call(D.new_method_call("org.bluez", "/", "org.freedesktop.DBus.ObjectManager", "GetManagedObjects"), 5000);
    DBusMessageIter it, objs, obj, ifaces, iface, props;
    if (!r) return 0;
    if (D.iter_init(r, &it) && D.arg_type(&it) == T_ARRAY)
        for (D.recurse(&it, &objs); D.arg_type(&objs) == T_DICT; D.next(&objs)) {
            const char *path, *name;
            D.recurse(&objs, &obj);
            D.get_basic(&obj, &path);
            D.next(&obj);
            for (D.recurse(&obj, &ifaces); D.arg_type(&ifaces) == T_DICT; D.next(&ifaces)) {
                D.recurse(&ifaces, &iface);
                D.get_basic(&iface, &name);
                D.next(&iface);
                D.recurse(&iface, &props);
                fn(path, name, &props, ctx);
            }
        }
    D.unref(r);
    return 1;
}

// The string property key of props, or NULL
static const char *prop_string(DBusMessageIter *props, const char *key) {
    for (; D.arg_type(props) == T_DICT; D.next(props)) {
        DBusMessageIter e, v;
        const char *k, *s;
        D.recurse(props, &e);
        D.get_basic(&e, &k);
        D.next(&e);
        if (strcmp(k, key) || !variant(&e, &v) || D.arg_type(&v) != T_STRING) continue;
        D.get_basic(&v, &s);
        return s;
    }
    return NULL;
}

static void find_adapter(const char *path, const char *iface, DBusMessageIter *props, void *ctx) {
    (void)ctx;
    const char *addr;
    if (adapter[0] || strcmp(iface, "org.bluez.Adapter1") || !(addr = prop_string(props, "Address"))) return;
    if (parse_address(addr, host)) snprintf(adapter, sizeof adapter, "%s", path);
}

static void find_chars(const char *path, const char *iface, DBusMessageIter *props, void *ctx) {
    S2Link *l = ctx;
    size_t n = strlen(l->dev);
    const char *uuid;
    if (strncmp(path, l->dev, n) || path[n] != '/' || !(uuid = prop_string(props, "UUID"))) return;
    if (!strcmp(iface, "org.bluez.GattDescriptor1")) {
        if (!strcasecmp(uuid, S2_UUID_RATE_DESC)) snprintf(l->rate, sizeof l->rate, "%s", path);
        return;
    }
    if (strcmp(iface, "org.bluez.GattCharacteristic1")) return;
    char *dst = !strcasecmp(uuid, S2_UUID_CMD_WR) ? l->cmd : !strcasecmp(uuid, S2_UUID_CMD_RSP) ? l->rsp
              : !strcasecmp(uuid, S2_UUID_INPUT) ? l->input
              : !strcasecmp(uuid, S2_UUID_VIB_PRO) || !strcasecmp(uuid, S2_UUID_VIB_JOYCON_L) ||
                    !strcasecmp(uuid, S2_UUID_VIB_JOYCON_R) ? l->vib : NULL;
    if (dst) snprintf(dst, sizeof l->cmd, "%s", path);
}

// Device1.ServicesResolved of path, pumping events while BlueZ reads the controller's GATT table
static int services_resolved(const char *path, int ms) {
    for (uint64_t end = switch2_now_ns() + (uint64_t)ms * 1000000; switch2_now_ns() < end; pump(100)) {
        DBusMessage *m = D.new_method_call("org.bluez", path, "org.freedesktop.DBus.Properties", "Get");
        DBusMessageIter it, v;
        const char *iface = "org.bluez.Device1", *name = "ServicesResolved";
        unsigned on = 0;
        D.init_append(m, &it);
        D.append_basic(&it, T_STRING, &iface);
        D.append_basic(&it, T_STRING, &name);
        DBusMessage *r = call(m, 2000);
        if (!r) return 0;
        if (D.iter_init(r, &it) && variant(&it, &v) && D.arg_type(&v) == T_BOOL) D.get_basic(&v, &on);
        D.unref(r);
        if (on) return 1;
    }
    return 0;
}

static void give_up(const char *path, const char *why) {
    rt_log("input: Switch 2 controller %s: %s", path, why);
    call_void(path, "org.bluez.Device1", "Disconnect", 5000);
    int slot = 0;
    for (int i = 1; i < MAX_PENDING; i++)
        if (failed[i].until < failed[slot].until) slot = i;
    snprintf(failed[slot].path, sizeof failed[slot].path, "%s", path);
    failed[slot].until = switch2_now_ns() + 10000000000ull;
}

static void connect_one(const char *path, uint16_t pid, int use) {
    rt_log("input: Switch 2 controller %s (product 0x%04x) %s, connecting", path, pid,
           use == S2_ADV_PAIR ? "in Sync" : "woke toward this PC");
    if (!call_void(path, "org.bluez.Device1", "Connect", 20000)) return give_up(path, "the connection failed");
    if (!services_resolved(path, 10000)) return give_up(path, "its services were not read");
    S2Link *l = calloc(1, sizeof *l);
    if (!l) return give_up(path, "out of memory");
    snprintf(l->dev, sizeof l->dev, "%s", path);
    l->pid = pid, l->use = use;
    managed(find_chars, l);
    if (!l->cmd[0] || !l->rsp[0] || !l->input[0]) return free(l), give_up(path, "not the GATT service of a Switch 2 controller");
    if (!call_void(l->rsp, "org.bluez.GattCharacteristic1", "StartNotify", 5000)) return free(l), give_up(path, "no replies");
    l->next = links, links = l;
    l->s = switch2_new(l, &ops, pid, use, host);
    busy = l;
    int ok = l->s && switch2_handshake(l->s);
    busy = NULL;
    if (ok && !l->gone) return;
    drop(l);
    give_up(path, ok ? "it disconnected during the handshake" : "the handshake failed");
}

static void bt_main(void *arg) {
    (void)arg;
    if (!load_dbus()) return rt_log("input: Switch 2 controllers off: libdbus-1.so.3 not found");
    DBusError e;
    D.error_init(&e);
    bus = D.bus_get_private(BUS_SYSTEM, &e);
    if (!bus) {
        rt_log("input: Switch 2 controllers off: no system bus (%s)", e.message ? e.message : "?");
        return D.error_free(&e);
    }
    if (!managed(find_adapter, NULL) || !adapter[0]) return rt_log("input: Switch 2 controllers off: no Bluetooth adapter (BlueZ)");
    D.add_match(bus, "type='signal',sender='org.bluez',interface='org.freedesktop.DBus.Properties',member='PropertiesChanged'", NULL);
    D.add_match(bus, "type='signal',sender='org.bluez',interface='org.freedesktop.DBus.ObjectManager',member='InterfacesAdded'", NULL);
    // LE only, and every advertisement (a controller woken again repeats one BlueZ has already seen)
    DBusMessage *m = method(adapter, "org.bluez.Adapter1", "SetDiscoveryFilter");
    DBusMessageIter it, f;
    const char *le = "le";
    unsigned yes = 1;
    D.init_append(m, &it);
    D.open(&it, T_ARRAY, "{sv}", &f);
    dict_entry(&f, "Transport", T_STRING, &le);
    dict_entry(&f, "DuplicateData", T_BOOL, &yes);
    D.close(&it, &f);
    DBusMessage *r = call(m, 5000);
    if (r) D.unref(r);
    if (!call_void(adapter, "org.bluez.Adapter1", "StartDiscovery", 5000))
        return rt_log("input: Switch 2 controllers off: discovery refused on %s (is Bluetooth on?)", adapter);
    rt_log("input: Switch 2 controllers: looking for them on %s (%02X:%02X:%02X:%02X:%02X:%02X); hold Sync on one to pair it",
           adapter, host[5], host[4], host[3], host[2], host[1], host[0]);
    for (uint64_t next_tick = 0;;) {
        pump(5);
        uint64_t now = switch2_now_ns();
        if (now >= next_tick) switch2_tick(), next_tick = now + 15000000;
        if (!npending) continue;
        char path[160];
        snprintf(path, sizeof path, "%s", pending[0].path);
        uint16_t pid = pending[0].pid;
        int use = pending[0].use;
        memmove(pending, pending + 1, sizeof *pending * (size_t)--npending);
        connect_one(path, pid, use);
    }
}

void switch2_start(void) {
    trace = cfg_int("TOTK_SWITCH2") >= 2;
    if (cfg_int("TOTK_SWITCH2") && !plat_thread_spawn(bt_main, NULL, "switch2", 0))
        rt_log("input: Switch 2 controllers off: no thread");
}

#endif   // __linux__
