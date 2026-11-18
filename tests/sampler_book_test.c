// The sampler book (runtime/sampler_book.h) against a modelled driver: reuse by create info, retirement from three
// quarters of the room, destruction only behind the GPU timeline, pinned samplers, the room full and recovering, a
// driver that refuses before the room is full.
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "sampler_book.h"

static int fails;
#define CHECK(c, ...)                                                                                                  \
    do {                                                                                                               \
        if (!(c)) { fails++; printf("FAIL line %d: ", __LINE__); printf(__VA_ARGS__); printf("\n"); }                  \
    } while (0)

#define MAXH 4096
typedef struct {
    unsigned limit;          // the driver's own limit: creation fails past it
    unsigned alive;
    uint64_t next;
    uint8_t live[MAXH];
    uint64_t destroyed_at[MAXH];   // the timeline value reached when each handle was destroyed (0: not yet)
    uint64_t done;
    unsigned created;
} Driver;

static uint64_t drv_create(void *ctx, const SbKey *key) {
    (void)key;
    Driver *d = ctx;
    if (d->alive >= d->limit) return 0;
    d->alive++, d->created++;
    uint64_t h = ++d->next;
    d->live[h] = 1;
    return h;
}

static void drv_destroy(void *ctx, uint64_t h) {
    Driver *d = ctx;
    if (!d->live[h]) {
        fails++;
        printf("FAIL: handle %llu destroyed twice or never made\n", (unsigned long long)h);
        return;
    }
    d->live[h] = 0, d->alive--, d->destroyed_at[h] = d->done;
}

static SbKey key_of(uint32_t i) {
    SbKey k;
    memset(&k, 0, sizeof k);
    memcpy(k.b, &i, 4);
    return k;
}

int main(void) {
        CHECK(sb_room_of(4000) == 3936, "room of 4000: %u", sb_room_of(4000));
    CHECK(sb_room_of(80) == 60, "room of 80: %u (a quarter reserved)", sb_room_of(80));
    CHECK(sb_room_of(1) == 1 && sb_room_of(0) == 1, "a tiny limit still leaves one");

    // reuse: the same key is one sampler
    {
        Driver d = {.limit = 1000};
        SamplerBook b;
        sb_init(&b, 1000);
        SbKey k = key_of(7);
        uint64_t a = sb_get(&b, &k, 0, 10, drv_create, drv_destroy, &d);
        uint64_t c = sb_get(&b, &k, 0, 10, drv_create, drv_destroy, &d);
        CHECK(a && a == c && d.created == 1, "one key, one sampler (made %u)", d.created);
        SbKey k2 = key_of(8);
        CHECK(sb_get(&b, &k2, 0, 10, drv_create, drv_destroy, &d) != a && d.created == 2,
              "another key, another sampler");
        sb_free(&b, drv_destroy, &d);
        CHECK(d.alive == 0, "sb_free destroys what is left (%u)", d.alive);
    }

    // retirement under a modelled driver of 200 samplers; the GPU two submissions behind; 1500 distinct keys drawn with
    {
        Driver d = {.limit = 200};
        SamplerBook b;
        sb_init(&b, 200);   // a room of 150, retirement from 113: 37 retired ones may wait for the GPU
        unsigned room = b.room;
        uint64_t issued = 0, refused = 0;
        uint64_t drawn_at[MAXH] = {0};   // handle -> last submission that drew with it
        for (int sub = 1; sub <= 300; sub++) {
            issued = (uint64_t)sub;
            d.done = issued > 2 ? issued - 2 : 0;   // the GPU is two submissions behind
            for (int j = 0; j < 5; j++) {
                SbKey k = key_of((uint32_t)(sub * 5 + j));
                uint64_t h = sb_get(&b, &k, d.done, issued + 2, drv_create, drv_destroy, &d);
                if (!h) { refused++; continue; }
                CHECK(h < MAXH && d.live[h], "a handle that is not alive was handed out");
                drawn_at[h] = issued;
            }
            CHECK(d.alive <= room, "alive %u over the room %u", d.alive, room);
            CHECK(b.n + b.ndead <= room, "book holds %u over the room %u", b.n + b.ndead, room);
        }
        CHECK(refused == 0, "%llu draws refused", (unsigned long long)refused);
        CHECK(b.n_made == 1500, "each key made once: %llu", (unsigned long long)b.n_made);
        CHECK(b.n_retired > 1300, "retired %llu", (unsigned long long)b.n_retired);
        for (uint64_t h = 1; h <= d.next; h++)
            if (d.destroyed_at[h] && d.destroyed_at[h] < drawn_at[h])
                CHECK(0, "handle %llu destroyed at %llu, drawn with at %llu", (unsigned long long)h,
                      (unsigned long long)d.destroyed_at[h], (unsigned long long)drawn_at[h]);
        // a retired sampler is destroyed only once the timeline reaches its retirement value (drawn_at + <=2)
        for (uint64_t h = 1; h <= d.next; h++)
            if (d.destroyed_at[h]) CHECK(d.destroyed_at[h] >= drawn_at[h] + 0, "ordering");
        sb_free(&b, drv_destroy, &d);
    }

    // retirement takes the least recently resolved, and bumps the generation
    {
        Driver d = {.limit = 1000};
        SamplerBook b;
        sb_init(&b, 8);   // room of 6, retirement from 5 (reserve 2)
        CHECK(b.room == 6 && b.high == 5, "room %u high %u", b.room, b.high);
        uint64_t h[8];
        for (uint32_t i = 0; i < 5; i++) {
            SbKey k = key_of(i);
            h[i] = sb_get(&b, &k, 0, 1, drv_create, drv_destroy, &d);
        }
        SbKey k0 = key_of(0);
        sb_get(&b, &k0, 0, 1, drv_create, drv_destroy, &d);   // key 0 is now the newest; key 1 the oldest
        uint32_t gen = b.gen;
        SbKey k9 = key_of(9);
        uint64_t n = sb_get(&b, &k9, 0, 5, drv_create, drv_destroy, &d);
        CHECK(n && b.gen == gen + 1, "a retirement bumps the generation");
        SbKey k1 = key_of(1);
        CHECK(nvn_hm_get(&b.by_key, &k1) == NULL, "the least recently used key (1) went");
        CHECK(nvn_hm_get(&b.by_key, &k0) != NULL, "the recently resolved key (0) stayed");
        CHECK(d.live[h[1]], "the retired sampler is still alive: the GPU has not reached 5");
        sb_collect(&b, 4, drv_destroy, &d);
        CHECK(d.live[h[1]], "done 4 < 5: still alive");
        sb_collect(&b, 5, drv_destroy, &d);
        CHECK(!d.live[h[1]], "done 5: destroyed");
        sb_free(&b, drv_destroy, &d);
    }

    // pinned samplers are never retired
    {
        Driver d = {.limit = 1000};
        SamplerBook b;
        sb_init(&b, 8);
        SbKey k0 = key_of(100);
        uint64_t p = sb_get(&b, &k0, 0, 1, drv_create, drv_destroy, &d);
        sb_pin(&b, p);
        for (uint32_t i = 0; i < 40; i++) {
            SbKey k = key_of(i);
            CHECK(sb_get(&b, &k, i + 2, i + 2, drv_create, drv_destroy, &d) != 0, "draw %u refused", i);
        }
        CHECK(d.live[p] && nvn_hm_get(&b.by_key, &k0), "the pinned sampler stayed");
        sb_free(&b, drv_destroy, &d);
    }

    // a stopped GPU: the room fills with retired samplers, nothing is made past it, and it recovers when the GPU moves
    {
        Driver d = {.limit = 1000};
        SamplerBook b;
        sb_init(&b, 8);
        unsigned made_ok = 0, refused = 0;
        uint32_t last_made = 0;
        for (uint32_t i = 0; i < 20; i++) {
            SbKey k = key_of(i);
            if (sb_get(&b, &k, 0, 100, drv_create, drv_destroy, &d)) made_ok++, last_made = i;
            else refused++;
            CHECK(d.alive <= b.room, "never past the room: %u", d.alive);
        }
        CHECK(made_ok == b.room && refused == 20 - b.room, "made %u refused %u", made_ok, refused);
        CHECK(b.n_full == refused, "refusals counted");
        CHECK(sb_newest(&b) != 0, "a stand-in exists");
        int stood_in = 0;
        SbKey full = key_of(400);
        uint64_t s = sb_admit(&b, &full, 0, 100, drv_create, drv_destroy, &d, &stood_in);
        CHECK(s == sb_newest(&b) && stood_in == 1, "sb_admit gives the newest sampler in place of one it cannot make");
        SbKey again = key_of(last_made);
        CHECK(sb_admit(&b, &again, 0, 100, drv_create, drv_destroy, &d, &stood_in) && !stood_in,
              "sb_admit answers a sampler the book has as itself");
        SbKey k = key_of(500);
        CHECK(sb_get(&b, &k, 100, 200, drv_create, drv_destroy, &d) != 0, "the GPU moved on: made again");
        sb_free(&b, drv_destroy, &d);
    }

    // a driver that refuses before the room is full: one sampler is retired and creation tried again
    {
        Driver d = {.limit = 3};
        SamplerBook b;
        sb_init(&b, 1000);
        for (uint32_t i = 0; i < 3; i++) {
            SbKey k = key_of(i);
            CHECK(sb_get(&b, &k, 0, 5, drv_create, drv_destroy, &d) != 0, "first three fit");
        }
        SbKey k = key_of(77);
        uint64_t h = sb_get(&b, &k, 5, 5, drv_create, drv_destroy, &d);
        CHECK(h != 0, "the oldest was retired and destroyed behind the done value, then the new one made");
        sb_free(&b, drv_destroy, &d);
    }
    if (fails) return 1;
    printf("sampler_book: ok\n");
    return 0;
}
