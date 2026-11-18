// The parts of platform.h built on its locks rather than on the OS: recursive mutexes, one-time initialisation and
// events. Every OS shares them.
#include "platform.h"

// ---- recursive mutex: owner is the holder's thread id (0 when free); only the holder writes depth.

void plat_rmutex_lock(PlatRMutex *m) {
    uint32_t me = plat_thread_id();
    if (__atomic_load_n(&m->owner, __ATOMIC_RELAXED) == me) {
        m->depth++;
        return;
    }
    plat_mutex_lock(&m->m);
    __atomic_store_n(&m->owner, me, __ATOMIC_RELAXED);
    m->depth = 1;
}

int plat_rmutex_trylock(PlatRMutex *m) {
    uint32_t me = plat_thread_id();
    if (__atomic_load_n(&m->owner, __ATOMIC_RELAXED) == me) {
        m->depth++;
        return 1;
    }
    if (!plat_mutex_trylock(&m->m)) return 0;
    __atomic_store_n(&m->owner, me, __ATOMIC_RELAXED);
    m->depth = 1;
    return 1;
}

void plat_rmutex_unlock(PlatRMutex *m) {
    if (--m->depth) return;
    __atomic_store_n(&m->owner, 0, __ATOMIC_RELAXED);
    plat_mutex_unlock(&m->m);
}

uint32_t plat_rmutex_release(PlatRMutex *m) {
    uint32_t depth = m->depth;
    m->depth = 1;
    plat_rmutex_unlock(m);
    return depth;
}

void plat_rmutex_retake(PlatRMutex *m, uint32_t depth) {
    plat_rmutex_lock(m);
    m->depth = depth;
}

// ---- once: state 0 not run, 1 running, 2 done. Waiters for a running initialiser sleep on one process-wide
// condition variable; initialisers are rare and short.

static PlatMutex once_lock = PLAT_MUTEX_INIT;
static PlatCond once_done = PLAT_COND_INIT;

void plat_once(PlatOnce *o, void (*fn)(void *arg), void *arg) {
    if (__atomic_load_n(&o->state, __ATOMIC_ACQUIRE) == 2) return;
    int32_t expected = 0;
    if (__atomic_compare_exchange_n(&o->state, &expected, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        fn(arg);
        plat_mutex_lock(&once_lock);
        __atomic_store_n(&o->state, 2, __ATOMIC_RELEASE);
        plat_cond_broadcast(&once_done);
        plat_mutex_unlock(&once_lock);
        return;
    }
    plat_mutex_lock(&once_lock);
    while (__atomic_load_n(&o->state, __ATOMIC_ACQUIRE) != 2) plat_cond_wait(&once_done, &once_lock);
    plat_mutex_unlock(&once_lock);
}

// ---- events

void plat_event_init(PlatEvent *e, int manual, int initially_set) {
    plat_mutex_init(&e->m);
    plat_cond_init(&e->c);
    e->manual = manual;
    e->set = initially_set;
}

void plat_event_set(PlatEvent *e) {
    plat_mutex_lock(&e->m);
    e->set = 1;
    if (e->manual) plat_cond_broadcast(&e->c);
    else plat_cond_signal(&e->c);
    plat_mutex_unlock(&e->m);
}

void plat_event_reset(PlatEvent *e) {
    plat_mutex_lock(&e->m);
    e->set = 0;
    plat_mutex_unlock(&e->m);
}

void plat_event_wait(PlatEvent *e) {
    plat_mutex_lock(&e->m);
    while (!e->set) plat_cond_wait(&e->c, &e->m);
    if (!e->manual) e->set = 0;
    plat_mutex_unlock(&e->m);
}

int plat_event_wait_ns(PlatEvent *e, uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    plat_mutex_lock(&e->m);
    while (!e->set) {
        uint64_t now = plat_now_ns();
        if (now >= end) break;
        plat_cond_wait_ns(&e->c, &e->m, end - now);
    }
    int got = e->set;
    if (got && !e->manual) e->set = 0;
    plat_mutex_unlock(&e->m);
    return got;
}

// ---- operations in flight (platform.h plat_op_begin)
// Each thread owns one slot of op_threads (claimed at its first operation, never given back: a thread that ends leaves
// an empty slot behind) and is the only writer of it; a reader copies a slot between two even reads of its seq, which
// the owner makes odd while it changes the slot. Finished operations go to op_ring, whose entries carry the index they
// were written at (plus one) once complete. Nothing here takes a lock: the system calls it brackets run in exception
// handlers (the lazy heap's fault handler) and under the runtime's locks.
#define OP_THREADS 1024
#define OP_DEPTH 8
#define OP_RING 16384   // a layer exit makes about 700 in one frame; a frame in play about 60
typedef struct {
    int64_t kind;
    uint64_t addr, size, t0;
} OpFrame;
typedef struct {
    volatile uint32_t seq;
    uint32_t tid;
    int32_t role, depth;
    OpFrame st[OP_DEPTH];
} OpThread;
typedef struct {
    volatile uint64_t seq;   // ring index + 1 once written, 0 while written
    uint32_t tid;
    int32_t role, kind, depth;
    uint64_t addr, size, run, t0, t1;
} OpRec;
int plat_op_on;
static OpThread op_threads[OP_THREADS];
static volatile int32_t op_nthreads;
static OpRec op_ring[OP_RING];
static volatile uint64_t op_head;
static volatile int64_t op_tot[PLAT_OP_ROLES][PLAT_OP_N][3];   // calls, ns, longest ns
static OpThread op_none;   // op_me of a thread that found every slot taken: it records nothing
static _Thread_local OpThread *op_me;
static _Thread_local int op_my_role;

void plat_op_trace(void) { __atomic_store_n(&plat_op_on, 1, __ATOMIC_RELEASE); }

void plat_op_role(int role) {
    op_my_role = role >= 0 && role < PLAT_OP_ROLES ? role : PLAT_OP_OTHER;
    if (op_me && op_me != &op_none) __atomic_store_n(&op_me->role, op_my_role, __ATOMIC_RELAXED);
}

const char *plat_op_name(int kind) {
    static const char *const name[] = {
        "readable query", "state query", "lazy heap query", "protect", "commit", "decommit", "write-watch query", "thaw",
        "fault thaw", "freeze", "freeze lock wait", "touch", "heap size", "pool init", "pool finalize",
        "pool release wait", "import", "import release", "file read"};
    _Static_assert(sizeof name / sizeof *name == PLAT_OP_N, "a name for every kind of operation");
    return kind >= 0 && kind < PLAT_OP_N ? name[kind] : "?";
}

const char *plat_op_role_name(int role) {
    static const char *const name[PLAT_OP_ROLES] = {"other host thread", "game thread", "render thread", "replay thread"};
    return role >= 0 && role < PLAT_OP_ROLES ? name[role] : "?";
}

static OpThread *op_self(void) {
    if (op_me) return op_me;
    int32_t i = __atomic_fetch_add(&op_nthreads, 1, __ATOMIC_RELAXED);
    if (i >= OP_THREADS) return op_me = &op_none;
    OpThread *t = &op_threads[i];
    __atomic_store_n(&t->tid, plat_thread_id(), __ATOMIC_RELAXED);
    __atomic_store_n(&t->role, op_my_role, __ATOMIC_RELAXED);
    __atomic_store_n(&t->depth, 0, __ATOMIC_RELEASE);
    return op_me = t;
}

int plat_op_begin_(int kind, uint64_t addr, uint64_t size) {
    OpThread *t = op_self();
    if (t == &op_none) return -1;
    int d = t->depth;
    uint64_t now = plat_now_ns();
    __atomic_store_n(&t->seq, t->seq + 1, __ATOMIC_RELAXED);   // odd: being changed
    __atomic_thread_fence(__ATOMIC_RELEASE);
    if (d < OP_DEPTH) {
        __atomic_store_n(&t->st[d].kind, kind, __ATOMIC_RELAXED);
        __atomic_store_n(&t->st[d].addr, addr, __ATOMIC_RELAXED);
        __atomic_store_n(&t->st[d].size, size, __ATOMIC_RELAXED);
        __atomic_store_n(&t->st[d].t0, now, __ATOMIC_RELAXED);
    }
    __atomic_store_n(&t->depth, d + 1, __ATOMIC_RELAXED);
    __atomic_store_n(&t->seq, t->seq + 1, __ATOMIC_RELEASE);
    return d;
}

static void op_max(volatile int64_t *m, int64_t v) {
    int64_t o = __atomic_load_n(m, __ATOMIC_RELAXED);
    while (v > o && !__atomic_compare_exchange_n(m, &o, v, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

void plat_op_end_(int token, uint64_t run) {
    OpThread *t = op_self();
    if (t == &op_none) return;
    int d = token;
    // An inner operation whose end was skipped is dropped with it (the token names the frame begin pushed); an end
    // whose frame is gone already does nothing.
    if (d < 0 || d >= t->depth) return;
    uint64_t now = plat_now_ns();
    OpFrame f = d < OP_DEPTH ? t->st[d] : (OpFrame){-1, 0, 0, now};
    __atomic_store_n(&t->seq, t->seq + 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&t->depth, d, __ATOMIC_RELAXED);
    __atomic_store_n(&t->seq, t->seq + 1, __ATOMIC_RELEASE);
    if (f.kind < 0 || f.kind >= PLAT_OP_N) return;   // nested deeper than OP_DEPTH: its start was not kept
    int role = __atomic_load_n(&t->role, __ATOMIC_RELAXED);
    int64_t ns = (int64_t)(now - f.t0);
    __atomic_add_fetch(&op_tot[role][f.kind][0], 1, __ATOMIC_RELAXED);
    __atomic_add_fetch(&op_tot[role][f.kind][1], ns, __ATOMIC_RELAXED);
    op_max(&op_tot[role][f.kind][2], ns);
    uint64_t i = __atomic_fetch_add(&op_head, 1, __ATOMIC_RELAXED);
    OpRec *r = &op_ring[i % OP_RING];
    __atomic_store_n(&r->seq, 0, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    __atomic_store_n(&r->tid, t->tid, __ATOMIC_RELAXED);
    __atomic_store_n(&r->role, role, __ATOMIC_RELAXED);
    __atomic_store_n(&r->kind, (int32_t)f.kind, __ATOMIC_RELAXED);
    __atomic_store_n(&r->depth, d, __ATOMIC_RELAXED);
    __atomic_store_n(&r->addr, f.addr, __ATOMIC_RELAXED);
    __atomic_store_n(&r->size, f.size, __ATOMIC_RELAXED);
    __atomic_store_n(&r->run, run, __ATOMIC_RELAXED);
    __atomic_store_n(&r->t0, f.t0, __ATOMIC_RELAXED);
    __atomic_store_n(&r->t1, now, __ATOMIC_RELAXED);
    __atomic_store_n(&r->seq, i + 1, __ATOMIC_RELEASE);
}

int plat_op_overlap(uint64_t t0, uint64_t t1, uint32_t skip_tid, PlatOp *out, int max, int *lost) {
    int n = 0;
    *lost = 0;
    uint64_t head = __atomic_load_n(&op_head, __ATOMIC_ACQUIRE);
    uint64_t oldest = head > OP_RING ? head - OP_RING : 0;
    for (uint64_t i = head; i > oldest && n < max; i--) {
        const OpRec *r = &op_ring[(i - 1) % OP_RING];
        uint64_t s = __atomic_load_n(&r->seq, __ATOMIC_ACQUIRE);
        if (s != i) continue;   // being written, or already overwritten by a newer one
        PlatOp o = {__atomic_load_n(&r->tid, __ATOMIC_RELAXED), __atomic_load_n(&r->role, __ATOMIC_RELAXED),
                    __atomic_load_n(&r->kind, __ATOMIC_RELAXED), __atomic_load_n(&r->depth, __ATOMIC_RELAXED), 1,
                    __atomic_load_n(&r->addr, __ATOMIC_RELAXED), __atomic_load_n(&r->size, __ATOMIC_RELAXED),
                    __atomic_load_n(&r->run, __ATOMIC_RELAXED), __atomic_load_n(&r->t0, __ATOMIC_RELAXED),
                    __atomic_load_n(&r->t1, __ATOMIC_RELAXED)};
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        if (__atomic_load_n(&r->seq, __ATOMIC_RELAXED) != s) continue;
        // Entries go in at their end, so one that ended a second before t0 has only older ones behind it (the second
        // covers a writer preempted between its end and its entry).
        if (o.t1 + 1000000000ull < t0) break;
        if (o.tid == skip_tid || o.t1 <= t0 || o.t0 >= t1) continue;
        out[n++] = o;
    }
    if (oldest) {   // the oldest entry still held ended after t0: older overlapping ones may have been overwritten
        const OpRec *r = &op_ring[oldest % OP_RING];
        if (__atomic_load_n(&r->seq, __ATOMIC_ACQUIRE) == oldest + 1 && __atomic_load_n(&r->t1, __ATOMIC_RELAXED) > t0)
            *lost = 1;
    }
    int nt = __atomic_load_n(&op_nthreads, __ATOMIC_RELAXED);
    if (nt > OP_THREADS) nt = OP_THREADS;
    uint64_t now = plat_now_ns();
    for (int k = 0; k < nt && n < max; k++) {
        const OpThread *t = &op_threads[k];
        uint32_t tid = __atomic_load_n(&t->tid, __ATOMIC_RELAXED);
        if (tid == skip_tid) continue;
        for (int tries = 0; tries < 4; tries++) {
            uint32_t s = __atomic_load_n(&t->seq, __ATOMIC_ACQUIRE);
            if (s & 1) continue;
            int d = __atomic_load_n(&t->depth, __ATOMIC_RELAXED), role = __atomic_load_n(&t->role, __ATOMIC_RELAXED);
            if (d > OP_DEPTH) d = OP_DEPTH;
            OpFrame f[OP_DEPTH];
            for (int j = 0; j < d; j++)
                f[j] = (OpFrame){__atomic_load_n(&t->st[j].kind, __ATOMIC_RELAXED),
                                 __atomic_load_n(&t->st[j].addr, __ATOMIC_RELAXED),
                                 __atomic_load_n(&t->st[j].size, __ATOMIC_RELAXED),
                                 __atomic_load_n(&t->st[j].t0, __ATOMIC_RELAXED)};
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
            if (__atomic_load_n(&t->seq, __ATOMIC_RELAXED) != s) continue;
            for (int j = 0; j < d && n < max; j++)
                if (f[j].t0 < t1 && f[j].kind >= 0 && f[j].kind < PLAT_OP_N)
                    out[n++] = (PlatOp){tid, role, (int)f[j].kind, j, 0, f[j].addr, f[j].size, 0, f[j].t0, now};
            break;
        }
    }
    return n;
}

void plat_op_totals(PlatOpTotal out[PLAT_OP_ROLES][PLAT_OP_N]) {
    for (int r = 0; r < PLAT_OP_ROLES; r++)
        for (int k = 0; k < PLAT_OP_N; k++) {
            out[r][k].calls = (uint64_t)__atomic_exchange_n(&op_tot[r][k][0], 0, __ATOMIC_RELAXED);
            out[r][k].ns = (uint64_t)__atomic_exchange_n(&op_tot[r][k][1], 0, __ATOMIC_RELAXED);
            out[r][k].max_ns = (uint64_t)__atomic_exchange_n(&op_tot[r][k][2], 0, __ATOMIC_RELAXED);
        }
}
