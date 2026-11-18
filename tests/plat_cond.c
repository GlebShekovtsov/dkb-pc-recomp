// Stress test of the Linux condition variables (runtime/platform/linux.c, plat_cond_*): a signal or broadcast skips
// the futex wake when no thread is counted as waiting, so a wake-up lost in the window between a waiter reading the
// sequence number and counting itself would leave a thread asleep for good. tests/plat_cond_test.py runs this under a
// timeout, which catches such a hang; the checks below catch a count that is left over.
#include "../runtime/platform/linux.c"

#include <pthread.h>

enum { ROUNDS = 200000, WAITERS = 8, GENERATIONS = 20000 };

static PlatMutex mu = PLAT_MUTEX_INIT;
static PlatCond turn_cv = PLAT_COND_INIT, gen_cv = PLAT_COND_INIT, done_cv = PLAT_COND_INIT;
static int turn;                  // ping-pong: whose move it is, 0 or 1
static unsigned gen, seen_last;   // broadcast: the generation, and how many waiters saw the last one

static void *ping_pong(void *arg) {
    int me = (int)(intptr_t)arg;
    for (int i = 0; i < ROUNDS; i++) {
        plat_mutex_lock(&mu);
        while (turn != me) plat_cond_wait(&turn_cv, &mu);
        turn = !me;
        if (i & 1) {   // signal under the lock, or after it as most callers in the runtime do
            plat_cond_signal(&turn_cv);
            plat_mutex_unlock(&mu);
        } else {
            plat_mutex_unlock(&mu);
            plat_cond_signal(&turn_cv);   // the other side is often not yet waiting: the wake is skipped then
        }
    }
    return NULL;
}

static void *follow(void *arg) {
    (void)arg;
    unsigned mine = 0;
    plat_mutex_lock(&mu);
    while (mine < GENERATIONS) {
        while (gen == mine) plat_cond_wait(&gen_cv, &mu);
        mine = gen;
    }
    seen_last++;
    plat_cond_signal(&done_cv);
    plat_mutex_unlock(&mu);
    return NULL;
}

static int fail(const char *what) {
    fprintf(stderr, "plat_cond: %s\n", what);
    return 1;
}

int main(void) {
    pthread_t t[WAITERS];

    // One thread each side, taking turns: every round has a signal that may find nobody waiting.
    pthread_create(&t[0], NULL, ping_pong, (void *)(intptr_t)0);
    pthread_create(&t[1], NULL, ping_pong, (void *)(intptr_t)1);
    pthread_join(t[0], NULL);
    pthread_join(t[1], NULL);
    if (*cv_waiters(&turn_cv)) return fail("ping-pong left a waiter counted");

    // Broadcasts to several waiters, with the generation moved on under the lock each time.
    for (int i = 0; i < WAITERS; i++) pthread_create(&t[i], NULL, follow, NULL);
    for (unsigned g = 1; g <= GENERATIONS; g++) {
        plat_mutex_lock(&mu);
        gen = g;
        plat_mutex_unlock(&mu);
        plat_cond_broadcast(&gen_cv);
    }
    plat_mutex_lock(&mu);
    while (seen_last < WAITERS) plat_cond_wait(&done_cv, &mu);
    plat_mutex_unlock(&mu);
    for (int i = 0; i < WAITERS; i++) pthread_join(t[i], NULL);
    if (*cv_waiters(&gen_cv) || *cv_waiters(&done_cv)) return fail("broadcast left a waiter counted");

    // A timed wait nobody signals times out and leaves no count behind.
    plat_mutex_lock(&mu);
    int woke = plat_cond_wait_ns(&turn_cv, &mu, 2000000);
    plat_mutex_unlock(&mu);
    if (woke) return fail("a timed wait with no signal did not time out");
    if (*cv_waiters(&turn_cv)) return fail("a timed-out wait left a waiter counted");

    printf("plat_cond: %d ping-pong rounds, %d broadcasts to %d waiters, timed wait: ok\n", ROUNDS, GENERATIONS,
           WAITERS);
    return 0;
}
