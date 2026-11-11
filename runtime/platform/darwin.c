// Apple silicon implementation of platform.h: pthreads and Mach threads, Mach virtual memory at fixed guest
// addresses, os_sync wait/wake in place of futexes, POSIX files, sysctl and Mach host statistics for the load
// counters, posix_spawn for child processes.
//
// Two things differ from linux.c in ways the rest of the runtime can see. Host pages are 16 KiB and guest pages
// 4 KiB, so a protection change is applied to the host pages it touches with the strongest permission of their guest
// pages (pstate keeps the guest state). And macOS has neither userfaultfd nor GetWriteWatch: writes into the watched
// reservations are not tracked, plat_mem_written reports them untracked, and the Vulkan back end uploads in full.
#define _DARWIN_C_SOURCE
#include <CoreFoundation/CoreFoundation.h>
#include <crt_externs.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/mach_vm.h>
#include <mach/thread_act.h>
#include <mach/thread_info.h>
#include <mach/thread_status.h>
#include <mach/vm_statistics.h>
#include <os/clock.h>
#include <os/os_sync_wait_on_address.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <SDL3/SDL_clipboard.h>

#include "platform.h"

extern char **environ;

// ---- threads

struct PlatThread {
    pthread_t t;
    PlatThreadFn fn;
    void *arg;
    volatile int32_t tid;   // the Mach thread port once the thread runs, for plat_thread_cpu_ns
    pthread_override_t qos;   // a priority set from outside, kept until the thread ends
    char name[16];
    // Completion for plat_thread_join and plat_thread_join_ns: macOS has no pthread_timedjoin_np.
    PlatMutex done_m;
    PlatCond done_c;
    int done;
};

// A signal stack for the calling thread, so the fault and crash handlers still run when its own stack is exhausted.
// Never freed: threads that end are few, and a stack freed while a handler might run on it would be worse.
static void alt_stack(void) {
    stack_t cur;
    if (!sigaltstack(NULL, &cur) && !(cur.ss_flags & SS_DISABLE)) return;
    size_t size = 64 << 10;
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return;
    stack_t ss = {.ss_sp = p, .ss_size = size};
    sigaltstack(&ss, NULL);
}

static void *thread_main(void *p) {
    PlatThread *t = p;
    alt_stack();
    __atomic_store_n(&t->tid, (int32_t)pthread_mach_thread_np(pthread_self()), __ATOMIC_RELEASE);
    if (t->name[0]) pthread_setname_np(t->name);   // one-argument form on macOS
    t->fn(t->arg);
    plat_mutex_lock(&t->done_m);
    t->done = 1;
    plat_cond_broadcast(&t->done_c);
    plat_mutex_unlock(&t->done_m);
    return NULL;
}

PlatThread *plat_thread_new(PlatThreadFn fn, void *arg, const char *name) {
    return plat_thread_new_ex(fn, arg, name, 0);
}

PlatThread *plat_thread_new_ex(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->fn = fn;
    t->arg = arg;
    if (name) snprintf(t->name, sizeof t->name, "%s", name);
    // The stack is mapped lazily, as Windows reserves it: only the pages a thread touches cost memory. Pthreads wants
    // whole host pages, at least PTHREAD_STACK_MIN.
    size_t sz = stack ? stack : PLAT_STACK_DEFAULT;
    sz = (sz + 16383) & ~(size_t)16383;
    pthread_attr_t a;
    pthread_attr_init(&a);
    pthread_attr_setstacksize(&a, sz);
    int e = pthread_create(&t->t, &a, thread_main, t);
    pthread_attr_destroy(&a);
    if (e) {
        free(t);
        return NULL;
    }
    return t;
}

// Apple silicon has no SMT: every logical processor is a physical core, so "one per core" is the whole set and the
// caller's ONE_PER_CORE knob stays off.
int plat_cpu_cores(uint64_t *all, uint64_t *first) {
    uint32_t n = plat_cpu_count();
    if (n > 64) n = 64;
    *all = n == 64 ? ~0ull : (1ull << n) - 1;
    *first = *all;
    uint64_t phys = 0;
    size_t sz = sizeof phys;
    if (sysctlbyname("hw.physicalcpu", &phys, &sz, NULL, 0) || !phys) return (int)n;
    return phys > (uint64_t)n ? (int)n : (int)phys;
}

uint32_t plat_cpu_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
}

uint32_t plat_cpu_usable(void) { return plat_cpu_count(); }

// Thread affinity does not exist on macOS (the scheduler places threads itself); both calls report failure, which the
// runtime already treats as "the OS chose".
uint64_t plat_thread_affinity(PlatThread *t, uint64_t mask) {
    (void)t, (void)mask;
    return 0;
}

int plat_process_affinity(uint64_t mask) {
    (void)mask;
    return 0;
}

// THREAD_BASIC_INFO gives the thread's user and system time in seconds and microseconds.
uint64_t plat_thread_cpu_ns(uint32_t tid) {
    thread_basic_info_data_t bi;
    mach_msg_type_number_t n = THREAD_BASIC_INFO_COUNT;
    if (thread_info((thread_act_t)tid, THREAD_BASIC_INFO, (thread_info_t)&bi, &n) != KERN_SUCCESS) return 0;
    return ((uint64_t)bi.user_time.seconds + (uint64_t)bi.system_time.seconds) * 1000000000u +
           ((uint64_t)bi.user_time.microseconds + (uint64_t)bi.system_time.microseconds) * 1000u;
}

// ---- thread interruption: Mach's suspend and thread state in place of Windows' SuspendThread/GetThreadContext.
// The thread port of a thread of this task carries the rights thread_suspend, thread_get_state and thread_set_state
// need, so the caller stops the thread where it runs, reads its pc and sp, redirects it and lets it go.

struct PlatIntr {
    mach_port_t thread;
    arm_thread_state64_t st;
};

PlatIntr *plat_intr_self(void) {
    PlatIntr *t = calloc(1, sizeof *t);
    if (!t) return NULL;
    t->thread = pthread_mach_thread_np(pthread_self());
    if (t->thread == MACH_PORT_NULL) {
        free(t);
        return NULL;
    }
    return t;
}

void plat_intr_free(PlatIntr *t) { free(t); }

int plat_intr_stop(PlatIntr *t, uint64_t *rip, uint64_t *rsp) {
    if (!t) return 0;
    if (thread_suspend(t->thread) != KERN_SUCCESS) return 0;
    mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
    if (thread_get_state(t->thread, ARM_THREAD_STATE64, (thread_state_t)&t->st, &n) != KERN_SUCCESS) {
        thread_resume(t->thread);
        return 0;
    }
    *rip = arm_thread_state64_get_pc(t->st);
    *rsp = arm_thread_state64_get_sp(t->st);
    return 1;
}

int plat_intr_resume(PlatIntr *t, uint64_t to) {
    if (!t) return 0;
    int ok = 1;
    if (to) {
        arm_thread_state64_set_pc_fptr(t->st, (void *)(uintptr_t)to);
        mach_msg_type_number_t n = ARM_THREAD_STATE64_COUNT;
        if (thread_set_state(t->thread, ARM_THREAD_STATE64, (thread_state_t)&t->st, n) != KERN_SUCCESS) ok = 0;
    }
    thread_resume(t->thread);
    return ok;
}

// plat_intr_return: the thread signals itself, and the handler writes the saved context over the one the signal
// interrupted, so sigreturn loads every register as it was when the thread was stopped. The signal is delivered before
// pthread_kill returns, on the thread's own stack below the saved context, which stays intact until the handler copied
// it. SIGUSR2 is not used otherwise.
#define INTR_RETURN_SIG SIGUSR2
static __thread const PlatIntrCtx *intr_ret;

static void intr_return_handler(int sig, siginfo_t *si, void *uc) {
    (void)sig, (void)si;
    const PlatIntrCtx *c = intr_ret;
    if (!c) return;
    intr_ret = NULL;
    ucontext_t *u = uc;
    for (int i = 0; i < 29; i++) u->uc_mcontext->__ss.__x[i] = c->x[i];
    arm_thread_state64_set_fp(u->uc_mcontext->__ss, c->x[29]);
    arm_thread_state64_set_lr_fptr(u->uc_mcontext->__ss, (void *)(uintptr_t)c->x[30]);
    arm_thread_state64_set_sp(u->uc_mcontext->__ss, c->sp);
    arm_thread_state64_set_pc_fptr(u->uc_mcontext->__ss, (void *)(uintptr_t)c->pc);
    u->uc_mcontext->__ss.__cpsr = (u->uc_mcontext->__ss.__cpsr & 0x0fffffffu) | ((uint32_t)c->nzcv & 0xf0000000u);
    memcpy(u->uc_mcontext->__ns.__v, c->q, sizeof u->uc_mcontext->__ns.__v);
    u->uc_mcontext->__ns.__fpsr = (uint32_t)c->fpsr;
    u->uc_mcontext->__ns.__fpcr = (uint32_t)c->fpcr;
}

static void intr_return_install(void) {
    struct sigaction sa = {.sa_sigaction = intr_return_handler, .sa_flags = SA_SIGINFO};
    sigemptyset(&sa.sa_mask);
    sigaction(INTR_RETURN_SIG, &sa, NULL);
}

void plat_intr_return(const PlatIntrCtx *c) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, intr_return_install);
    sigset_t s;
    sigemptyset(&s);
    sigaddset(&s, INTR_RETURN_SIG);
    pthread_sigmask(SIG_UNBLOCK, &s, NULL);
    intr_ret = c;
    pthread_kill(pthread_self(), INTR_RETURN_SIG);
    abort();   // not reached: the handler sent the thread back to c->pc
}

// ---- fibers: a stack switch of our own, as in linux.c. fiber_swap saves the callee-saved general and FP registers
// and FPCR/FPSR on the current stack, stores the stack pointer in *save and loads the other one. A new fiber's stack
// is laid out as fiber_swap leaves one, returning into fiber_entry, which calls fn(arg) (x19 and x20).
struct PlatFiber {
    void *sp;
    void *stack;   // mapped stack (NULL for a thread's own context)
    size_t size;
};

void plat_fiber_swap(void **save, void *load);
void plat_fiber_entry(void);
__asm__(".text\n"
        ".globl _plat_fiber_swap\n"
        ".p2align 4\n"
        "_plat_fiber_swap:\n"
        "    sub  sp, sp, #192\n"
        "    stp  d8, d9, [sp, #0]\n"
        "    stp  d10, d11, [sp, #16]\n"
        "    stp  d12, d13, [sp, #32]\n"
        "    stp  d14, d15, [sp, #48]\n"
        "    mrs  x2, fpcr\n"
        "    mrs  x3, fpsr\n"
        "    stp  x2, x3, [sp, #64]\n"
        "    stp  x19, x20, [sp, #80]\n"
        "    stp  x21, x22, [sp, #96]\n"
        "    stp  x23, x24, [sp, #112]\n"
        "    stp  x25, x26, [sp, #128]\n"
        "    stp  x27, x28, [sp, #144]\n"
        "    stp  x29, x30, [sp, #160]\n"
        "    str  xzr, [sp, #176]\n"
        "    mov  x2, sp\n"
        "    str  x2, [x0]\n"
        "    mov  sp, x1\n"
        "    ldp  x2, x3, [sp, #64]\n"
        "    msr  fpcr, x2\n"
        "    msr  fpsr, x3\n"
        "    ldp  d8, d9, [sp, #0]\n"
        "    ldp  d10, d11, [sp, #16]\n"
        "    ldp  d12, d13, [sp, #32]\n"
        "    ldp  d14, d15, [sp, #48]\n"
        "    ldp  x19, x20, [sp, #80]\n"
        "    ldp  x21, x22, [sp, #96]\n"
        "    ldp  x23, x24, [sp, #112]\n"
        "    ldp  x25, x26, [sp, #128]\n"
        "    ldp  x27, x28, [sp, #144]\n"
        "    ldp  x29, x30, [sp, #160]\n"
        "    add  sp, sp, #192\n"
        "    ret\n"
        ".globl _plat_fiber_entry\n"
        ".p2align 4\n"
        "_plat_fiber_entry:\n"
        "    mov  x0, x20\n"
        "    blr  x19\n"
        "    brk  #0\n");

static __thread PlatFiber *fiber_cur;
static __thread PlatFiber fiber_own;

PlatFiber *plat_fiber_self(void) {
    fiber_own = (PlatFiber){0};
    fiber_cur = &fiber_own;
    return &fiber_own;
}

void plat_fiber_unself(void) { fiber_cur = NULL; }

PlatFiber *plat_fiber_new(size_t stack, void (*fn)(void *arg), void *arg) {
    PlatFiber *f = calloc(1, sizeof *f);
    if (!f) return NULL;
    // Reserved, not committed; the lowest host page stays inaccessible, so an overflow faults instead of corrupting.
    void *m = mmap(NULL, stack, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) {
        free(f);
        return NULL;
    }
    mprotect(m, 16384, PROT_NONE);
    f->stack = m, f->size = stack;
    uint64_t *top = (uint64_t *)(((uintptr_t)m + stack) & ~(uintptr_t)15);
    uint64_t *fr = top - 24;   // d8..d15, FPCR, FPSR, x19..x28, x29, x30, padding
    memset(fr, 0, 192);
    fr[10] = (uint64_t)(uintptr_t)fn;                   // x19
    fr[11] = (uint64_t)(uintptr_t)arg;                  // x20
    fr[21] = (uint64_t)(uintptr_t)plat_fiber_entry;     // x30
    f->sp = fr;
    return f;
}

void plat_fiber_switch(PlatFiber *to) {
    PlatFiber *from = fiber_cur;
    fiber_cur = to;   // on this thread; the switch back into from sets it on whichever thread that is
    plat_fiber_swap(&from->sp, to->sp);
}

void plat_fiber_free(PlatFiber *f) {
    if (!f) return;
    if (f->stack) munmap(f->stack, f->size);
    free(f);
}

int plat_thread_spawn(PlatThreadFn fn, void *arg, const char *name, size_t stack) {
    PlatThread *t = plat_thread_new_ex(fn, arg, name, stack);
    if (t) plat_thread_detach(t);
    return t != NULL;
}

void plat_thread_join(PlatThread *t) {
    plat_mutex_lock(&t->done_m);
    while (!t->done) plat_cond_wait(&t->done_c, &t->done_m);
    plat_mutex_unlock(&t->done_m);
    pthread_join(t->t, NULL);
    free(t);
}

int plat_thread_join_ns(PlatThread *t, uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    plat_mutex_lock(&t->done_m);
    while (!t->done) {
        uint64_t now = plat_now_ns();
        if (now >= end || !plat_cond_wait_ns(&t->done_c, &t->done_m, end - now)) break;
    }
    int done = t->done;
    plat_mutex_unlock(&t->done_m);
    if (!done) return 0;
    pthread_join(t->t, NULL);
    free(t);
    return 1;
}

// The thread keeps using t->fn and t->arg, so the struct stays allocated (a few bytes per detached thread).
void plat_thread_detach(PlatThread *t) { pthread_detach(t->t); }

// QoS classes stand in for Windows' seven levels. A thread with no class of its own inherits from the thread that
// created it, and every class here is one this runtime sets explicitly, so the mapping back is unambiguous.
static int qos_of(int prio) {
    return prio <= PLAT_PRIO_IDLE ? QOS_CLASS_BACKGROUND
         : prio <= PLAT_PRIO_BELOW_NORMAL ? QOS_CLASS_UTILITY
         : prio == PLAT_PRIO_NORMAL ? QOS_CLASS_DEFAULT
         : prio <= PLAT_PRIO_HIGHEST ? QOS_CLASS_USER_INITIATED
         : QOS_CLASS_USER_INTERACTIVE;
}

void plat_thread_priority(int prio) { pthread_set_qos_class_self_np((qos_class_t)qos_of(prio), 0); }

// A class set on another thread is a QOS override, the form macOS offers for a thread that is already running. An
// override only raises a thread above its own class: a lower priority ends the previous override and leaves the
// thread at its own class, which is as low as another thread can put it.
void plat_thread_set_priority(PlatThread *t, int prio) {
    if (!t) return;
    if (t->qos) pthread_override_qos_class_end_np(t->qos);
    t->qos = NULL;
    qos_class_t own = QOS_CLASS_UNSPECIFIED;
    int rel = 0;
    if (pthread_get_qos_class_np(t->t, &own, &rel)) own = QOS_CLASS_UNSPECIFIED;
    qos_class_t want = (qos_class_t)qos_of(prio);
    if (want > own) t->qos = pthread_override_qos_class_start_np(t->t, want, 0);
}

int plat_thread_priority_get(void) {
    qos_class_t q = QOS_CLASS_DEFAULT;
    int rel = 0;
    pthread_get_qos_class_np(pthread_self(), &q, &rel);
    switch (q) {
    case QOS_CLASS_BACKGROUND: return PLAT_PRIO_IDLE;
    case QOS_CLASS_UTILITY: return PLAT_PRIO_BELOW_NORMAL;
    case QOS_CLASS_USER_INITIATED: return PLAT_PRIO_ABOVE_NORMAL;
    case QOS_CLASS_USER_INTERACTIVE: return PLAT_PRIO_TIME_CRITICAL;
    default: return PLAT_PRIO_NORMAL;
    }
}

void plat_yield(void) { sched_yield(); }

// ---- time

uint64_t plat_now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

uint64_t plat_counter(void) { return plat_now_ns(); }
uint64_t plat_counter_hz(void) { return 1000000000u; }

// macOS's timers are accurate to well under a millisecond without asking.
void plat_timer_1ms(void) {}

// nanosleep has no absolute form here; the last PLAT_SPIN_NS are spent yielding so the deadline is met closely.
#define PLAT_SPIN_NS 100000u

void plat_sleep_ns(uint64_t ns) {
    uint64_t end = plat_now_ns() + ns;
    while (plat_now_ns() + PLAT_SPIN_NS < end) {
        uint64_t left = end - PLAT_SPIN_NS - plat_now_ns();
        struct timespec ts = {(time_t)(left / 1000000000u), (long)(left % 1000000000u)};
        nanosleep(&ts, NULL);
    }
    while (plat_now_ns() < end) sched_yield();
}

void plat_wait_ns(uint64_t ns) {
    struct timespec ts = {(time_t)(ns / 1000000000u), (long)(ns % 1000000000u)};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

void plat_sleep_ms(uint32_t ms) {
    struct timespec ts = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000};
    while (nanosleep(&ts, &ts) == -1 && errno == EINTR) {}
}

// ---- locks: the same 32-bit reader-writer word as linux.c.
// bit 31: a writer holds it; bit 30: some thread sleeps on the word; bits 0-29: readers holding it.

#define LK_W 0x80000000u
#define LK_S 0x40000000u
#define LK_R 0x3fffffffu
#define LK_SPINS 64

// os_sync_wait_on_address is Darwin's futex: it compares the 4 bytes at w with value and sleeps while they are equal.
// It can return early (EINTR, or EFAULT under memory pressure) with the value still unchanged, so every caller tests
// the value again after it returns, as the header recommends.
static void sync_wait(volatile uint32_t *w, uint32_t value) {
    while (os_sync_wait_on_address((void *)w, value, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE) < 0 &&
           (errno == EINTR || errno == EFAULT))
        ;
}

static int sync_wait_ns(volatile uint32_t *w, uint32_t value, uint64_t ns) {
    errno = 0;
    int r = os_sync_wait_on_address_with_timeout((void *)w, value, 4, OS_SYNC_WAIT_ON_ADDRESS_NONE,
                                                 OS_CLOCK_MACH_ABSOLUTE_TIME, ns);
    return r >= 0 || errno != ETIMEDOUT;
}

static inline volatile uint32_t *lk(PlatMutex *m) { return (volatile uint32_t *)&m->p; }

// Sets the sleeper bit on s and sleeps while the word still equals the result.
static void lk_sleep(volatile uint32_t *w, uint32_t s) {
    if (!(s & LK_S) && !__atomic_compare_exchange_n(w, &s, s | LK_S, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) return;
    sync_wait(w, s | LK_S);
}

static void lk_wake(volatile uint32_t *w) {
    __atomic_fetch_and(w, ~LK_S, __ATOMIC_RELAXED);
    os_sync_wake_by_address_all((void *)w, 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

void plat_mutex_init(PlatMutex *m) { m->p = 0; }

int plat_mutex_trylock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_R)))
        if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_R))) {
            if (__atomic_compare_exchange_n(w, &s, s | LK_W, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_fetch_and(w, ~LK_W, __ATOMIC_RELEASE);
    if (s & LK_S) lk_wake(w);
}

int plat_mutex_trylock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
    while (!(s & (LK_W | LK_S)))
        if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return 1;
    return 0;
}

void plat_mutex_lock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    for (int spin = 0;; spin++) {
        uint32_t s = __atomic_load_n(w, __ATOMIC_RELAXED);
        if (!(s & (LK_W | LK_S))) {
            if (__atomic_compare_exchange_n(w, &s, s + 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
            continue;
        }
        if (spin < LK_SPINS) plat_pause();
        else lk_sleep(w, s);
    }
}

void plat_mutex_unlock_shared(PlatMutex *m) {
    volatile uint32_t *w = lk(m);
    uint32_t s = __atomic_sub_fetch(w, 1, __ATOMIC_RELEASE);
    if (!(s & LK_R) && (s & LK_S)) lk_wake(w);
}

// ---- condition variables: a sequence number in the first 32 bits, as in linux.c.

static inline volatile uint32_t *cv(PlatCond *c) { return (volatile uint32_t *)&c->p; }

void plat_cond_init(PlatCond *c) { c->p = 0; }

void plat_cond_wait(PlatCond *c, PlatMutex *m) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    plat_mutex_unlock(m);
    sync_wait(cv(c), seq);
    plat_mutex_lock(m);
}

int plat_cond_wait_ns(PlatCond *c, PlatMutex *m, uint64_t ns) {
    uint32_t seq = __atomic_load_n(cv(c), __ATOMIC_RELAXED);
    plat_mutex_unlock(m);
    int woke = sync_wait_ns(cv(c), seq, ns);
    plat_mutex_lock(m);
    return woke;
}

void plat_cond_signal(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    os_sync_wake_by_address_any((void *)cv(c), 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

void plat_cond_broadcast(PlatCond *c) {
    __atomic_add_fetch(cv(c), 1, __ATOMIC_RELEASE);
    os_sync_wake_by_address_all((void *)cv(c), 4, OS_SYNC_WAKE_BY_ADDRESS_NONE);
}

// ---- files

int plat_file_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && !S_ISDIR(st.st_mode);
}

int plat_dir_exists(const char *path) {
    struct stat st;
    return !stat(path, &st) && S_ISDIR(st.st_mode);
}

int plat_full_path(const char *path, char *out, size_t cap) {
    if (path[0] == '/') return snprintf(out, cap, "%s", path) < (int)cap;
    char cwd[PATH_MAX];
    if (!getcwd(cwd, sizeof cwd)) return 0;
    return snprintf(out, cap, "%s/%s", cwd, path) < (int)cap;
}

void plat_file_mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) return;
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/' || *p == '\\') {
            char c = *p;
            *p = 0;
            mkdir(tmp, 0755);
            *p = c;
        }
    }
    mkdir(tmp, 0755);
}

static int write_all(int fd, const void *data, size_t n) {
    const uint8_t *p = data;
    while (n) {
        ssize_t k = write(fd, p, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return 0;
        p += k, n -= (size_t)k;
    }
    return 1;
}

uint8_t *plat_file_read(const char *path, uint64_t *size) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    uint8_t *buf = NULL;
    if (!fstat(fd, &st) && S_ISREG(st.st_mode) && st.st_size > 0 && st.st_size < (1ll << 30)) {
        buf = malloc((size_t)st.st_size + 1);
        size_t got = 0;
        while (buf && got < (size_t)st.st_size) {
            ssize_t k = read(fd, buf + got, (size_t)st.st_size - got);
            if (k < 0 && errno == EINTR) continue;
            if (k <= 0) break;
            got += (size_t)k;
        }
        if (buf && got != (size_t)st.st_size) {
            free(buf);
            buf = NULL;
        }
        if (buf) {
            buf[st.st_size] = 0;
            *size = (uint64_t)st.st_size;
        }
    }
    close(fd);
    return buf;
}

int plat_file_write(const char *path, const void *data, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n);
    return close(fd) == 0 && ok;
}

// O_APPEND makes each write land whole at the end of the file, from any thread.
PlatAppend *plat_append_open(const char *path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
    return fd < 0 ? NULL : (PlatAppend *)(intptr_t)(fd + 1);
}

int plat_append_write(PlatAppend *f, const void *data, size_t n) {
    int fd = (int)(intptr_t)f - 1;
    ssize_t k;
    do k = write(fd, data, n);
    while (k < 0 && errno == EINTR);
    return k == (ssize_t)n;
}

// Flushes the directory holding path, so a rename into it survives a crash.
static void sync_dir_of(const char *path) {
    char dir[PATH_MAX];
    if (snprintf(dir, sizeof dir, "%s", path) >= (int)sizeof dir) return;
    char *slash = strrchr(dir, '/');
    if (slash == dir) slash[1] = 0;
    else if (slash) *slash = 0;
    else strcpy(dir, ".");
    int fd = open(dir, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

int plat_file_replace(const char *from, const char *to) {
    if (rename(from, to)) return 0;
    sync_dir_of(to);
    return 1;
}

int plat_file_write_atomic(const char *path, const void *data, size_t n) {
    // A rename replaces a read-only file here (only the folder's permission counts) but not on Windows: a file the
    // player made read-only stays as it is on both.
    if (access(path, W_OK) && errno != ENOENT) return 0;
    char tmp[PATH_MAX + 48];
    int len = snprintf(tmp, sizeof tmp, "%s.%u.%u.tmp", path, plat_process_id(), plat_thread_id());
    if (len < 0 || len >= (int)sizeof tmp) return 0;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n) && fsync(fd) == 0;
    ok = close(fd) == 0 && ok;
    if (ok && !rename(tmp, path)) {
        sync_dir_of(path);
        return 1;
    }
    unlink(tmp);
    return 0;
}

// See platform.h: replace and write_atomic for re-derivable cache files, without the flush to the disk.
int plat_file_replace_cache(const char *from, const char *to) { return rename(from, to) == 0; }

int plat_file_write_cache(const char *path, const void *data, size_t n) {
    if (access(path, W_OK) && errno != ENOENT) return 0;
    char tmp[PATH_MAX + 48];
    int len = snprintf(tmp, sizeof tmp, "%s.%u.%u.tmp", path, plat_process_id(), plat_thread_id());
    if (len < 0 || len >= (int)sizeof tmp) return 0;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return 0;
    int ok = write_all(fd, data, n);
    ok = close(fd) == 0 && ok;
    if (ok && !rename(tmp, path)) return 1;
    unlink(tmp);
    return 0;
}

int plat_file_flush(const char *path) {
    int fd = open(path, O_WRONLY | O_CLOEXEC);
    if (fd < 0) return 0;
    int ok = fsync(fd) == 0;
    close(fd);
    return ok;
}

int plat_file_sync(FILE *f) { return !fflush(f) && !fsync(fileno(f)); }

void plat_file_delete(const char *path) { unlink(path); }

int plat_file_truncate(const char *path, uint64_t size) { return truncate(path, (off_t)size) == 0; }

int64_t plat_file_size(const char *path) {
    struct stat st;
    return stat(path, &st) ? -1 : (int64_t)st.st_size;
}

// The owner's write permission stands for Windows' read-only attribute.
int plat_file_set_readonly(const char *path, int on) {
    struct stat st;
    if (stat(path, &st)) return 0;
    mode_t m = on ? st.st_mode & ~(mode_t)0222 : st.st_mode | S_IWUSR;
    return chmod(path, m & 07777) == 0;
}

int plat_file_list(const char *dir, const char *ext, int (*cb)(const char *name, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return 0;
    size_t el = strlen(ext);
    for (struct dirent *e; (e = readdir(d));) {
        size_t n = strlen(e->d_name);
        if (n < el || strcmp(e->d_name + n - el, ext) || e->d_name[0] == '.') continue;
        if (!cb(e->d_name, ctx)) break;
    }
    closedir(d);
    return 1;
}

// ---- virtual memory. Guest pages are 4 KiB, macOS host pages are 16 KiB: pstate records one byte per guest page
// (PS_RESERVED or a committed permission) and a change is applied to whole host pages with the strongest permission
// of their guest pages, so nothing that another guest page still uses gets sealed.
//
// Reservations are fixed-address Mach allocations, made inaccessible; commit and protect change pstate and the host
// protection. Decommit gives a whole host page back with mach_vm_deallocate + mach_vm_allocate (a fresh allocation
// reads zero) and zeroes the partly covered host pages at the ends with memset. macOS has no write watch: a watched
// reservation is reserved as any other and plat_mem_written reports its pages untracked.

#define PG 4096ull
#define HPG 16384ull
#define PS_LIMIT (1ull << 39)
enum { PS_FREE, PS_RESERVED, PS_NONE, PS_READ, PS_RW };
static uint8_t *pstate;
static PlatMutex vm_lock = PLAT_MUTEX_INIT;
static int ww_warned;

static int vm_init(void) {   // vm_lock held
    if (pstate) return 1;
    void *p = mmap(NULL, (size_t)(PS_LIMIT / PG), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    pstate = p == MAP_FAILED ? NULL : p;   // anonymous memory reads zero, which is PS_FREE
    return pstate != NULL;
}

static int host_prot(uint64_t h) {
    int best = PLAT_PROT_NONE;
    for (uint64_t g = h; g < h + HPG; g += PG) {
        uint8_t v = pstate[g / PG];
        int p = v == PS_RW ? PLAT_PROT_RW : v == PS_READ ? PLAT_PROT_READ : PLAT_PROT_NONE;
        if (p > best) best = p;
    }
    return best == PLAT_PROT_RW ? VM_PROT_READ | VM_PROT_WRITE : best == PLAT_PROT_READ ? VM_PROT_READ : VM_PROT_NONE;
}

// Sets the guest pages [addr, addr + n) to v and makes the host pages they touch match, one call per run of host pages
// that ends up with the same permission. vm_lock held.
static int ps_apply(uint64_t addr, uint64_t n, uint8_t v) {
    if (!pstate || addr >= PS_LIMIT || n > PS_LIMIT - addr) return 0;
    n = (n + PG - 1) & ~(PG - 1);
    memset(pstate + addr / PG, v, (size_t)(n / PG));
    uint64_t h0 = addr & ~(HPG - 1), h1 = (addr + n + HPG - 1) & ~(HPG - 1);
    if (h1 > PS_LIMIT) h1 = PS_LIMIT;
    for (uint64_t h = h0; h < h1;) {
        int p = host_prot(h);
        uint64_t e = h + HPG;
        while (e < h1 && host_prot(e) == p) e += HPG;
        if (mach_vm_protect(mach_task_self(), h, e - h, FALSE, (vm_prot_t)p)) return 0;
        h = e;
    }
    return 1;
}

int plat_mem_reserve(uint64_t addr, uint64_t n, int watch) {
    if (!n || (addr | n) & (PG - 1)) return 0;
    plat_mutex_lock(&vm_lock);
    int ok = 0;
    if (vm_init() && addr + n <= PS_LIMIT) {
        if (watch && !ww_warned) {
            ww_warned = 1;
            fprintf(stderr, "platform: macOS has no write watch (no userfaultfd): written memory is not tracked, "
                            "texture uploads run in full\n");
        }
        mach_vm_address_t a = addr;
        kern_return_t kr = mach_vm_allocate(mach_task_self(), &a, n, VM_FLAGS_FIXED);
        if (kr == KERN_SUCCESS && a != addr) {
            mach_vm_deallocate(mach_task_self(), a, n);   // a fixed allocation stays put; never keep one elsewhere
        } else if (kr == KERN_SUCCESS) {
            if (!mach_vm_protect(mach_task_self(), addr, n, FALSE, VM_PROT_NONE)) {
                ok = ps_apply(addr, n, PS_RESERVED);
                if (!ok) mach_vm_deallocate(mach_task_self(), addr, n);
            } else {
                mach_vm_deallocate(mach_task_self(), addr, n);
            }
        }
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

// The lowest hole of [lo, hi) that holds size bytes at an align boundary and is not mapped by anyone (our own
// reservations included: mach_vm_region lists them). *at gets the candidate; 0 when none fits. vm_lock held.
static int find_hole(uint64_t lo, uint64_t hi, uint64_t size, uint64_t align, uint64_t *at) {
    uint64_t a = lo;
    while (a < hi) {
        mach_vm_address_t q = a;
        mach_vm_size_t sz = 0;
        vm_region_extended_info_data_t e;
        mach_msg_type_number_t cnt = VM_REGION_EXTENDED_INFO_COUNT;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t kr = mach_vm_region(mach_task_self(), &q, &sz, VM_REGION_EXTENDED_INFO, (vm_region_info_t)&e,
                                          &cnt, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        uint64_t limit = kr == KERN_SUCCESS ? q : hi;
        if (limit > hi) limit = hi;
        if (limit > a) {
            uint64_t c = (a + align - 1) & ~(align - 1);
            if (c >= a && c <= limit && size <= limit - c) {
                *at = c;
                return 1;
            }
        }
        if (kr != KERN_SUCCESS || q + sz > hi) break;
        a = q + sz;
    }
    return 0;
}

// Says how the window above the modules is actually laid out when a reservation finds no room there: the size asked
// for and the largest hole that is left. Runs without vm_lock (a diagnostic: the list moving under it is harmless),
// and only for TOTK_MEM_HOLES, since memory.c steps down through several sizes on a normal boot.
static void hole_report(uint64_t lo, uint64_t hi, uint64_t size) {
    uint64_t best = 0, best_at = 0, a = lo;
    while (a < hi) {
        mach_vm_address_t q = a;
        mach_vm_size_t sz = 0;
        vm_region_extended_info_data_t e;
        mach_msg_type_number_t cnt = VM_REGION_EXTENDED_INFO_COUNT;
        mach_port_t object = MACH_PORT_NULL;
        kern_return_t kr = mach_vm_region(mach_task_self(), &q, &sz, VM_REGION_EXTENDED_INFO, (vm_region_info_t)&e,
                                          &cnt, &object);
        if (object != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), object);
        uint64_t limit = kr == KERN_SUCCESS ? q : hi;
        if (limit > hi) limit = hi;
        if (limit > a && limit - a > best) best = limit - a, best_at = a;
        if (kr != KERN_SUCCESS || q + sz > hi) break;
        a = q + sz;
    }
    if (hi > a && hi - a > best) best = hi - a, best_at = a;
    if (!getenv("TOTK_MEM_HOLES")) return;
    fprintf(stderr, "platform: no %llu MiB hole above %#llx up to %#llx (the largest left is %llu MiB at %#llx)\n",
            (unsigned long long)(size >> 20), (unsigned long long)lo, (unsigned long long)hi,
            (unsigned long long)(best >> 20), (unsigned long long)best_at);
}

int plat_mem_reserve_in(uint64_t lo, uint64_t hi, uint64_t size, uint64_t align, int watch, uint64_t *at) {
    if (!size || hi <= lo || size > hi - lo) return 0;
    if (!align) align = PG;
    if (align & (align - 1)) return 0;
    uint64_t a = lo;
    while (a + size <= hi) {
        uint64_t c;
        plat_mutex_lock(&vm_lock);
        int found = find_hole(a, hi, size, align, &c);
        plat_mutex_unlock(&vm_lock);
        if (!found) {
            hole_report(a, hi, size);
            return 0;
        }
        if (plat_mem_reserve(c, size, watch)) {
            *at = c;
            return 1;
        }
        a = c + align;   // another thread took it between the walk and the reservation
    }
    return 0;
}

void *plat_mem_alloc(size_t n) {
    void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

int plat_mem_commit(uint64_t addr, size_t n) {
    if (!n) return 1;
    plat_mutex_lock(&vm_lock);
    int ok = ps_apply(addr, n, PS_RW);
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_decommit(uint64_t addr, uint64_t n) {
    if (!n) return 1;
    plat_mutex_lock(&vm_lock);
    int ok = 0;
    if (pstate && addr < PS_LIMIT && n <= PS_LIMIT - addr) {
        uint64_t end = addr + n;
        // Whole host pages inside the range go back to the system: a fixed allocation over them that replaces the
        // mapping in one call (VM_FLAGS_OVERWRITE) reads zero, and leaves no moment in which the range is unmapped and
        // another thread's mapping could land there. The partly covered host pages at the ends keep the pages outside
        // the range, so they are zeroed in place.
        uint64_t h0 = (addr + HPG - 1) & ~(HPG - 1), h1 = end & ~(HPG - 1);
        ok = 1;
        if (h1 > h0) {
            mach_vm_address_t a = h0;
            if (mach_vm_allocate(mach_task_self(), &a, h1 - h0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE) != KERN_SUCCESS ||
                a != h0)
                ok = 0;
        }
        if (ok && (addr < h0 || end > h1)) {
            uint64_t e0 = addr & ~(HPG - 1), e1 = (end + HPG - 1) & ~(HPG - 1);
            if (e1 > PS_LIMIT) e1 = PS_LIMIT;
            if (mach_vm_protect(mach_task_self(), e0, e1 - e0, FALSE, VM_PROT_READ | VM_PROT_WRITE)) ok = 0;
            if (ok && addr < h0) memset((void *)(uintptr_t)addr, 0, (size_t)((h0 < end ? h0 : end) - addr));
            if (ok && end > h1) memset((void *)(uintptr_t)(h1 > addr ? h1 : addr), 0, (size_t)(end - (h1 > addr ? h1 : addr)));
        }
        if (ok) ok = ps_apply(addr, n, PS_RESERVED);
    }
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_protect(uint64_t addr, uint64_t n, int prot) {
    if (!n) return 1;
    plat_mutex_lock(&vm_lock);
    int ok = ps_apply(addr, n, (uint8_t)(PS_NONE + prot));
    plat_mutex_unlock(&vm_lock);
    return ok;
}

int plat_mem_state(uint64_t addr, uint64_t limit, uint64_t *end) {
    if (!pstate || addr >= PS_LIMIT) {
        *end = limit;
        return PLAT_MEM_FREE;
    }
    uint64_t lim = limit < PS_LIMIT ? limit : PS_LIMIT, i = addr / PG;
    uint8_t v = pstate[i];
    while ((i + 1) * PG < lim && pstate[i + 1] == v) i++;
    *end = (i + 1) * PG < lim ? (i + 1) * PG : lim;
    return v == PS_FREE ? PLAT_MEM_FREE : v == PS_RESERVED ? PLAT_MEM_RESERVED : v - PS_NONE;
}

// No userfaultfd, no PAGEMAP_SCAN: the pages the CPU wrote since the last reset cannot be read back. The caller
// (vk_memory.c) uploads the whole range again when it sees this.
uint64_t plat_mem_written(uint64_t addr, uint64_t n, uint64_t *pages, uint64_t max, int reset) {
    (void)addr, (void)n, (void)pages, (void)max, (void)reset;
    return UINT64_MAX;
}

// One handler for the synchronous faults. A data abort whose syndrome says write goes to the write-fault hook first
// (memory.c thaws a frozen region); a fault in this executable's code then goes to the crash hook (main.c); anything
// else gets the default action. Runs on the thread's signal stack.
static int (*fault_hook)(uint64_t addr);
static void (*crash_fn)(const PlatCrash *c);

static void fault_handler(int sig, siginfo_t *si, void *uc) {
    ucontext_t *u = uc;
    uint64_t esr = (uint64_t)u->uc_mcontext->__es.__esr;
    unsigned ec = (unsigned)((esr >> 26) & 0x3f);   // 0x24/0x25 data abort, 0x20/0x21 instruction abort
    // XNU reports a write to a read-only page as SIGBUS (an access-protection fault on a mapped page), not SIGSEGV.
    if ((sig == SIGSEGV || sig == SIGBUS) && fault_hook && (ec == 0x24 || ec == 0x25) && (esr & (1u << 6)) &&
        fault_hook((uint64_t)(uintptr_t)si->si_addr))
        return;
    uintptr_t rip = (uintptr_t)arm_thread_state64_get_pc(u->uc_mcontext->__ss);
    Dl_info at, self;
    if (crash_fn && dladdr((void *)rip, &at) && dladdr((void *)plat_crash_hook, &self) && at.dli_fbase == self.dli_fbase) {
        PlatCrash c = {.what = sig == SIGSEGV ? "access violation" : sig == SIGBUS ? "bus error"
                        : sig == SIGILL ? "illegal instruction" : "arithmetic exception",
                       .code = (uint32_t)sig, .rip = rip, .offset = rip - (uintptr_t)self.dli_fbase, .access = -1};
        if (sig == SIGSEGV || sig == SIGBUS) {
            c.access = ec == 0x20 || ec == 0x21 ? 8 : (esr & (1u << 6)) ? 1 : 0;
            c.addr = (uint64_t)(uintptr_t)si->si_addr;
        }
        crash_fn(&c);
    }
    signal(sig, SIG_DFL);   // returns into the faulting instruction, which faults again with the default action
}

static void fault_install(void) {
    static int done;
    if (done) return;
    done = 1;
    alt_stack();
    struct sigaction sa = {.sa_sigaction = fault_handler, .sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER};
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}

void plat_mem_fault_hook(int (*fn)(uint64_t addr)) {
    fault_hook = fn;
    fault_install();
}

void plat_crash_hook(void (*fn)(const PlatCrash *c)) {
    crash_fn = fn;
    fault_install();
}

_Noreturn void plat_reexec(void) {
    char exe[PATH_MAX];
    fflush(stdout);
    fflush(stderr);
    if (plat_exe_path(exe, sizeof exe)) execv(exe, *_NSGetArgv());
    fprintf(stderr, "FATAL: re-running the process failed (%s)\n", strerror(errno));
    _exit(70);
}

int plat_mem_read(uint64_t addr, void *out, size_t n) {
    if (!n) return 1;
    mach_vm_size_t got = 0;
    if (mach_vm_read_overwrite(mach_task_self(), addr, n, (mach_vm_address_t)(uintptr_t)out, &got)) return 0;
    return got == n;
}

int plat_mem_readable(uint64_t addr, uint64_t n) {
    uint8_t b;
    uint64_t end = addr + (n ? n : 1);
    for (uint64_t a = addr; a < end; a = (a | 4095) + 1)
        if (!plat_mem_read(a, &b, 1)) return 0;
    return 1;
}

void *plat_aligned_alloc(size_t size, size_t align) {
    void *p = NULL;
    return posix_memalign(&p, align < sizeof(void *) ? sizeof(void *) : align, size) ? NULL : p;
}

void plat_aligned_free(void *p) { free(p); }

// A PlatFile is the descriptor + 1, so descriptor 0 is not NULL.
PlatFile *plat_file_open(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

PlatFile *plat_file_open_unbuffered(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return NULL;
    fcntl(fd, F_NOCACHE, 1);
    return (PlatFile *)(intptr_t)(fd + 1);
}

static int fd_of(PlatFile *f) { return (int)(intptr_t)f - 1; }

void plat_file_close(PlatFile *f) {
    if (f) close(fd_of(f));
}

int64_t plat_file_pread(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pread(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k >= 0 ? k : errno == EFAULT ? -2 : -1;
}

int plat_file_id(PlatFile *f, PlatFileId *out) {
    struct stat st;
    if (fstat(fd_of(f), &st)) return 0;
    *out = (PlatFileId){(uint64_t)st.st_dev, (uint64_t)st.st_ino, (uint64_t)st.st_size,
                        (uint64_t)st.st_mtimespec.tv_sec * 1000000000u + (uint64_t)st.st_mtimespec.tv_nsec,
                        (uint64_t)st.st_ctimespec.tv_sec * 1000000000u + (uint64_t)st.st_ctimespec.tv_nsec};
    return 1;
}

const void *plat_file_map(PlatFile *f) {
    struct stat st;
    if (fstat(fd_of(f), &st) || !st.st_size) return NULL;
    void *v = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd_of(f), 0);
    return v == MAP_FAILED ? NULL : v;
}

static int fs_err(int e) {
    switch (e) {
    case ENOENT: case ENOTDIR: return PLAT_FS_NOT_FOUND;
    case EEXIST: return PLAT_FS_EXISTS;
    case ENOTEMPTY: return PLAT_FS_NOT_EMPTY;
    case EBUSY: case ETXTBSY: return PLAT_FS_BUSY;
    default: return PLAT_FS_ERROR;
    }
}

int plat_fs_kind(const char *path) {
    struct stat st;
    if (lstat(path, &st)) return PLAT_FS_MISSING;
    return S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
}

int plat_fs_mkdir(const char *path) { return mkdir(path, 0755) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_rmdir(const char *path) {
    struct stat st;
    if (!lstat(path, &st) && S_ISLNK(st.st_mode)) return unlink(path) ? fs_err(errno) : PLAT_FS_OK;
    return rmdir(path) ? fs_err(errno) : PLAT_FS_OK;
}

// Deleting needs write access to the directory, not to the file: a read-only file goes as any other.
int plat_fs_unlink(const char *path) { return unlink(path) ? fs_err(errno) : PLAT_FS_OK; }

// RENAME_EXCL keeps Windows' rule that a rename never replaces; a volume without it gets a check first, which another
// process could race but the save directories are ours.
int plat_fs_rename(const char *from, const char *to, int durable) {
    if (renamex_np(from, to, RENAME_EXCL)) {
        if (errno != EINVAL && errno != ENOTSUP) return fs_err(errno);
        if (plat_fs_kind(to) != PLAT_FS_MISSING) return PLAT_FS_EXISTS;
        if (rename(from, to)) return fs_err(errno);
    }
    if (durable) {
        sync_dir_of(from);
        sync_dir_of(to);
    }
    return PLAT_FS_OK;
}

int plat_fs_link(const char *existing, const char *path) { return link(existing, path) ? fs_err(errno) : PLAT_FS_OK; }

int plat_fs_links(const char *path) {
    struct stat st;
    return stat(path, &st) ? 0 : (int)st.st_nlink;
}

void plat_fs_sync_dir(const char *path) {
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
}

// Open files never stop a rename here.
void plat_fs_busy_reason(const char *dir, char *out, size_t cap) {
    (void)dir;
    if (cap) *out = 0;
}

int plat_fs_list(const char *dir, int (*cb)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    DIR *d = opendir(dir);
    if (!d) return fs_err(errno);
    for (struct dirent *e; (e = readdir(d));) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (fstatat(dirfd(d), e->d_name, &st, AT_SYMLINK_NOFOLLOW)) continue;
        int kind = S_ISLNK(st.st_mode) ? PLAT_FS_LINK : S_ISDIR(st.st_mode) ? PLAT_FS_DIR : PLAT_FS_FILE;
        if (!cb(e->d_name, kind, kind == PLAT_FS_FILE ? (uint64_t)st.st_size : 0, ctx)) break;
    }
    closedir(d);
    return PLAT_FS_OK;
}

PlatFile *plat_fs_open(const char *path, int write, int *err) {
    struct stat st;
    int fd = open(path, (write ? O_RDWR : O_RDONLY) | O_CLOEXEC);
    if (fd >= 0 && (fstat(fd, &st) || !S_ISREG(st.st_mode))) {
        close(fd);
        fd = -1, errno = ENOENT;
    }
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

PlatFile *plat_fs_create(const char *path, int *err) {
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0 && err) *err = fs_err(errno);
    return fd < 0 ? NULL : (PlatFile *)(intptr_t)(fd + 1);
}

int64_t plat_file_pwrite(PlatFile *f, const void *buf, uint32_t n, uint64_t off) {
    ssize_t k;
    do k = pwrite(fd_of(f), buf, n, (off_t)off);
    while (k < 0 && errno == EINTR);
    return k;
}

int plat_file_set_size(PlatFile *f, uint64_t size) { return ftruncate(fd_of(f), (off_t)size) ? fs_err(errno) : 0; }

int plat_file_fsync(PlatFile *f) { return fsync(fd_of(f)) == 0; }

int plat_temp_dir(char *out, size_t cap) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = "/tmp";
    size_t n = strlen(t);
    while (n > 1 && t[n - 1] == '/') n--;
    return snprintf(out, cap, "%.*s", (int)n, t) < (int)cap;
}

uintptr_t plat_exe_base(void) {
    Dl_info d;
    return dladdr((void *)plat_exe_base, &d) ? (uintptr_t)d.dli_fbase : 0;
}

int plat_module_of(uintptr_t addr, char *name, size_t cap, uintptr_t *base) {
    Dl_info d;
    if (!dladdr((void *)addr, &d) || !d.dli_fname) return 0;
    const char *n = strrchr(d.dli_fname, '/');
    *base = (uintptr_t)d.dli_fbase;
    return snprintf(name, cap, "%s", n ? n + 1 : d.dli_fname) < (int)cap;
}

int plat_exe_path(char *out, size_t cap) {
    char tmp[PATH_MAX];
    uint32_t size = sizeof tmp;
    if (_NSGetExecutablePath(tmp, &size)) return 0;
    char real[PATH_MAX];
    const char *p = realpath(tmp, real) ? real : tmp;
    return snprintf(out, cap, "%s", p) < (int)cap;
}

size_t plat_getenv(const char *name, char *out, size_t cap) {
    const char *v = getenv(name);
    size_t n = v ? strlen(v) : 0;
    if (!n || n >= cap) return 0;
    memcpy(out, v, n + 1);
    return n;
}

// A thread's Mach port name is unique in the task while the thread lives, and the kernel keeps it a valid port for
// thread_info (plat_thread_cpu_ns). Cached per thread: plat_rmutex_lock asks on every acquire.
uint32_t plat_thread_id(void) {
    static __thread uint32_t id;
    if (!id) id = (uint32_t)pthread_mach_thread_np(pthread_self());
    return id;
}

// The first preferred language is the UI language; the current locale gives the regional format.
const char *plat_ui_language(void) {
    static char tag[64];
    if (!tag[0]) {
        CFArrayRef langs = CFLocaleCopyPreferredLanguages();
        if (langs) {
            if (CFArrayGetCount(langs) > 0) {
                CFStringRef s = (CFStringRef)CFArrayGetValueAtIndex(langs, 0);
                if (s) CFStringGetCString(s, tag, sizeof tag, kCFStringEncodingUTF8);
            }
            CFRelease(langs);
        }
    }
    return tag;
}

const char *plat_user_locale(void) {
    static char tag[64];
    if (!tag[0]) {
        CFLocaleRef l = CFLocaleCopyCurrent();
        if (l) {
            CFStringRef s = CFLocaleGetIdentifier(l);
            if (s) CFStringGetCString(s, tag, sizeof tag, kCFStringEncodingUTF8);
            CFRelease(l);
        }
    }
    return tag;
}

// The system's CJK fonts: each covers its language's script, and every macOS install has them.
int plat_font_path(const char *lang, char *out, size_t cap) {
    static const struct { const char *lang, *path; } files[] = {
        {"ja", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"},
        {"ko", "/System/Library/Fonts/AppleSDGothicNeo.ttc"},
        {"zh-Hans", "/System/Library/Fonts/PingFang.ttc"},
        {"zh-Hant", "/System/Library/Fonts/PingFang.ttc"},
    };
    for (size_t i = 0; i < sizeof files / sizeof *files; i++)
        if (!strcmp(lang, files[i].lang) && plat_file_exists(files[i].path))
            return snprintf(out, cap, "%s", files[i].path) < (int)cap;
    return 0;
}

// "macOS 27.0.1 (26A434)".
int plat_os_version(char *out, size_t cap) {
    char ver[64] = "", build[64] = "";
    size_t n = sizeof ver;
    if (sysctlbyname("kern.osproductversion", ver, &n, NULL, 0)) ver[0] = 0;
    n = sizeof build;
    if (sysctlbyname("kern.osversion", build, &n, NULL, 0)) build[0] = 0;
    if (ver[0] && build[0]) return snprintf(out, cap, "macOS %s (%s)", ver, build) < (int)cap;
    if (ver[0]) return snprintf(out, cap, "macOS %s", ver) < (int)cap;
    return 0;
}

int plat_stderr_path(char *out, size_t cap) {
    struct stat st;
    if (fstat(2, &st) || !S_ISREG(st.st_mode)) return 0;
    char path[PATH_MAX];
    if (fcntl(2, F_GETPATH, path) == -1) return 0;
    return snprintf(out, cap, "%s", path) < (int)cap;
}

// ---- process

void plat_exit(int code) {
    fflush(stdout);
    fflush(stderr);
    _exit(code);
}

int plat_setenv(const char *name, const char *value) {
    return (value ? setenv(name, value, 1) : unsetenv(name)) == 0;
}

uint32_t plat_process_id(void) { return (uint32_t)getpid(); }

uint64_t plat_process_cpu_ns(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

// PROC_PIDTBSDINFO carries the wall-clock start time; the uptime is the time since then.
uint64_t plat_process_uptime_ns(void) {
    struct proc_bsdinfo pb;
    if (proc_pidinfo(getpid(), PROC_PIDTBSDINFO, 0, &pb, sizeof pb) != (int)sizeof pb) return 0;
    uint64_t start = (uint64_t)pb.pbi_start_tvsec * 1000000000u + (uint64_t)pb.pbi_start_tvusec * 1000u;
    uint64_t now = plat_unix_time_ns();
    return now > start ? now - start : 0;
}

uint64_t plat_unix_time_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

void plat_local_time(struct tm *out) {
    time_t t = time(NULL);
    localtime_r(&t, out);
}

// The range of the Windows CRT's localtime_s and gmtime_s (1970 to 3000), which macOS exceeds: the game's time
// service reports times outside it as invalid on both hosts.
#define PLAT_TIME_MAX 32535215999ll

int plat_localtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && localtime_r(&t, out) != NULL;
}

int plat_gmtime(int64_t unix_s, struct tm *out) {
    time_t t = (time_t)unix_s;
    return unix_s >= 0 && unix_s <= PLAT_TIME_MAX && gmtime_r(&t, out) != NULL;
}

long plat_tz_west(void) {
    tzset();
    return timezone;
}

static void cloexec(int fd) { fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC); }

int plat_run_self(const char *env_name, const char *env_value, char *out, size_t cap, uint32_t timeout_ms) {
    char exe[PATH_MAX];
    if (!plat_exe_path(exe, sizeof exe)) return -1;
    int p[2];
    if (pipe(p)) return -1;
    cloexec(p[0]);
    cloexec(p[1]);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_adddup2(&fa, p[1], 2);
    // The child's environment is a copy with env_name set: setenv here could race getenv on other threads.
    size_t ne = 0, nl = strlen(env_name);
    while (environ[ne]) ne++;
    char **envp = calloc(ne + 2, sizeof *envp);
    char *entry = malloc(nl + strlen(env_value) + 2);
    pid_t pid;
    int e = -1;
    if (envp && entry) {
        sprintf(entry, "%s=%s", env_name, env_value);
        size_t k = 0;
        for (size_t i = 0; i < ne; i++)
            if (strncmp(environ[i], env_name, nl) || environ[i][nl] != '=') envp[k++] = environ[i];
        envp[k++] = entry;
        e = posix_spawn(&pid, exe, &fa, NULL, *_NSGetArgv(), envp);
    }
    free(envp);
    free(entry);
    posix_spawn_file_actions_destroy(&fa);
    close(p[1]);
    if (e) {
        close(p[0]);
        return -1;
    }
    size_t len = 0;
    char sink[512];
    ssize_t k;
    while (len + 1 < cap && (k = read(p[0], out + len, cap - 1 - len)) != 0)
        if (k > 0) len += (size_t)k;
        else if (errno != EINTR) break;
    if (cap) out[len] = 0;
    while ((k = read(p[0], sink, sizeof sink)) > 0 || (k < 0 && errno == EINTR)) {}
    close(p[0]);
    int status = 0;
    uint64_t end = plat_now_ns() + (uint64_t)timeout_ms * 1000000u;
    while (waitpid(pid, &status, timeout_ms ? WNOHANG : 0) == 0) {
        if (plat_now_ns() >= end) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            break;
        }
        plat_sleep_ms(10);
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
}

int plat_random(void *out, size_t n) {
    arc4random_buf(out, n);
    return 1;
}

// ---- desktop

int plat_clipboard_set(const char *utf8) { return SDL_SetClipboardText(utf8); }

static void reap_main(void *arg) { waitpid((pid_t)(intptr_t)arg, NULL, 0); }

// "open" hands the folder to the Finder; it runs detached, so a slow answer does not hold up the caller, and a thread
// of its own reaps it so it does not stay a zombie.
int plat_open_folder(const char *path) {
    struct stat st;
    if (stat(path, &st) || !S_ISDIR(st.st_mode)) return 0;
    char *argv[] = {"/usr/bin/open", (char *)path, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, "/usr/bin/open", NULL, NULL, argv, environ);
    if (e) return 0;
    plat_thread_spawn(reap_main, (void *)(intptr_t)pid, "reap", 64 << 10);
    return 1;
}

// ---- dynamic libraries

void *plat_lib_open(const char *name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void *plat_lib_sym(void *lib, const char *name) { return lib ? dlsym(lib, name) : NULL; }

// ---- machine load: Mach host statistics for the CPU and memory, sysctl for the installed memory. macOS exposes no
// GPU engine counters and no CPU temperature to an unprivileged process, so those stay -1.

struct PlatLoad {
    uint64_t busy, total;   // the previous call's CPU ticks; total 0 before the first
    uint64_t proc_ns, wall_ns;
};

PlatLoad *plat_load_open(const uint8_t *gpu_luid, int pci_bus) {
    (void)gpu_luid, (void)pci_bus;
    return calloc(1, sizeof(PlatLoad));
}

void plat_load_close(PlatLoad *l) { free(l); }

void plat_cpu_load(PlatLoad *l, float *total_pct, float *proc_pct, float *temp_c) {
    *total_pct = *proc_pct = *temp_c = -1;
    if (!l) return;
    host_cpu_load_info_data_t ci;
    mach_msg_type_number_t n = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t)&ci, &n) == KERN_SUCCESS) {
        uint64_t total = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) total += ci.cpu_ticks[i];
        uint64_t busy = total - ci.cpu_ticks[CPU_STATE_IDLE];
        uint64_t pns = plat_process_cpu_ns(), now = plat_now_ns();
        if (l->total && total > l->total && now > l->wall_ns) {
            double dt = (double)(total - l->total), dt_ns = (double)(now - l->wall_ns);
            *total_pct = (float)(100.0 * (double)(busy - l->busy) / dt);
            *proc_pct = (float)(100.0 * (double)(pns - l->proc_ns) / dt_ns / (double)plat_cpu_count());
        }
        l->busy = busy, l->total = total, l->proc_ns = pns, l->wall_ns = now;
    }
}

void plat_gpu_load(PlatLoad *l, float *load_pct, float *vram_used_mb) {
    (void)l;
    *load_pct = *vram_used_mb = -1;
}

void plat_mem_info(float *used_mb, float *total_mb, float *proc_mb) {
    *used_mb = *total_mb = *proc_mb = -1;
    uint64_t total = 0;
    size_t sz = sizeof total;
    if (!sysctlbyname("hw.memsize", &total, &sz, NULL, 0) && total) {
        vm_statistics64_data_t vm;
        mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &n) == KERN_SUCCESS) {
            uint64_t used = (uint64_t)(vm.active_count + vm.wire_count + vm.compressor_page_count) * vm_page_size;
            *total_mb = (float)(total >> 20);
            *used_mb = (float)(used >> 20);
        }
    }
    uint64_t resident, private_bytes;
    plat_process_memory(&resident, &private_bytes);
    if (resident) *proc_mb = (float)(resident >> 20);
}

// No walk of the address space on this host: mem_commit_report then sums nothing.
void plat_mem_walk(void (*fn)(uint64_t, uint64_t, int, void *), void *arg) { (void)fn, (void)arg; }
// Not implemented on this host (the commit report is Windows only): reports 0.
uint64_t plat_mem_resident(uint64_t base, uint64_t size) { (void)base, (void)size; return 0; }

void plat_process_memory(uint64_t *resident, uint64_t *private_bytes) {
    *resident = *private_bytes = 0;
    mach_task_basic_info_data_t bi;
    mach_msg_type_number_t n = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&bi, &n) == KERN_SUCCESS) *resident = bi.resident_size;
    task_vm_info_data_t vi;
    n = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) == KERN_SUCCESS)
        *private_bytes = vi.phys_footprint;
}

// macOS has no commit limit and no global commit counter; the process's footprint still makes sense.
void plat_commit_info(double *proc_mb, double *used_mb, double *limit_mb) {
    *proc_mb = *used_mb = *limit_mb = -1;
    uint64_t resident, private_bytes;
    plat_process_memory(&resident, &private_bytes);
    if (private_bytes) *proc_mb = (double)(private_bytes >> 20);
}

// ---- child processes

struct PlatChild {
    pid_t pid;
    int in, out;
};

PlatChild *plat_child_start(const char *exe, const char *arg, uint32_t *err) {
    if (err) *err = 0;
    int in[2], out[2];
    if (pipe(in)) return NULL;
    cloexec(in[0]);
    cloexec(in[1]);
    if (pipe(out)) {
        close(in[0]), close(in[1]);
        return NULL;
    }
    cloexec(out[0]);
    cloexec(out[1]);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    char *argv[] = {(char *)exe, (char *)arg, NULL};
    pid_t pid;
    int e = posix_spawn(&pid, exe, &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(in[0]);
    close(out[1]);
    if (e) {
        close(in[1]), close(out[0]);
        if (err) *err = (uint32_t)e;
        return NULL;
    }
    PlatChild *c = malloc(sizeof *c);
    if (!c) {
        kill(pid, SIGKILL);
        waitpid(pid, NULL, 0);
        close(in[1]), close(out[0]);
        return NULL;
    }
    c->pid = pid, c->in = in[1], c->out = out[0];
    return c;
}

void plat_child_stop(PlatChild *c) {
    if (!c) return;
    close(c->in);
    close(c->out);
    kill(c->pid, SIGKILL);
    waitpid(c->pid, NULL, 0);
    free(c);
}

int plat_child_write(PlatChild *c, const void *data, size_t n) { return write_all(c->in, data, n); }

long plat_child_read(PlatChild *c, void *buf, size_t cap) {
    ssize_t k;
    do k = read(c->out, buf, cap);
    while (k < 0 && errno == EINTR);
    return (long)k;
}
