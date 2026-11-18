// The Linux memory-pressure rule (runtime/linux_pressure.h): the meminfo reading, the shortfall and the episode gate.
#include <stdio.h>
#include <stdlib.h>

#include "linux_pressure.h"

#define MB (1ull << 20)

static int failures;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); failures++; } } while (0)

static const char *MEMINFO =
    "MemTotal:       16384000 kB\nMemFree:          100000 kB\nMemAvailable:    2097152 kB\nBuffers: 1 kB\n"
    "SwapTotal:       8388608 kB\nSwapFree:        4194304 kB\n";

int main(void) {
    uint64_t av, sw, tot;
    int has;
    // parsing
    CHECK(lp_parse_meminfo(MEMINFO, &av, &sw, &tot, &has));
    CHECK(av == 2048 * MB && sw == 4096 * MB && tot == 16000 * MB && has == 1);
    // no MemAvailable: nothing to compare
    CHECK(!lp_parse_meminfo("MemTotal: 5 kB\nSwapFree: 7 kB\n", &av, &sw, NULL, NULL));
    CHECK(lp_parse_meminfo("MemAvailable: 100 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n", &av, &sw, NULL, &has) &&
          has == 0);
    CHECK(lp_parse_meminfo("MemAvailable: 100 kB\n", &av, &sw, NULL, &has) && has == 1);   // no swap line: stricter
    CHECK(lp_parse_meminfo("MemAvailable: 100 kB", &av, &sw, NULL, NULL) && av == 100 * 1024);   // no final newline
    CHECK(lp_parse_meminfo(NULL, &av, &sw, NULL, NULL) == 0);

    // reserves from settings
    CHECK(lp_reserve_bytes(NULL, 1536) == 1536 * MB);
    CHECK(lp_reserve_bytes("", 1536) == 1536 * MB);
    CHECK(lp_reserve_bytes("0", 1536) == 0);
    CHECK(lp_reserve_bytes("2048", 1536) == 2048 * MB);
    CHECK(lp_reserve_bytes("12x", 1536) == 1536 * MB);
    CHECK(lp_reserve_bytes("-5", 1536) == 1536 * MB);
    CHECK(lp_reserve_bytes("99999999999999999999999", 1536) == 1536 * MB);

    // shortfall: the larger of the physical and the commit one
    const uint64_t A = 1536 * MB, C = 2560 * MB;
    CHECK(lp_shortfall(2048 * MB, 4096 * MB, A, C, 1) == 0);
    CHECK(lp_shortfall(1000 * MB, 8000 * MB, A, C, 1) == 536 * MB);          // physical only
    CHECK(lp_shortfall(2000 * MB, 100 * MB, A, C, 1) == 460 * MB);           // commit only
    CHECK(lp_shortfall(1000 * MB, 100 * MB, A, C, 1) == 1460 * MB);          // both: the larger (commit 1460 > 536)
    CHECK(lp_shortfall(2000 * MB, 100 * MB, A, C, 0) == 0);                  // no swap: the commit check is off
    CHECK(lp_shortfall(100 * MB, 0, 0, 0, 1) == 0);                          // both reserves off
    CHECK(lp_shortfall(100 * MB, 0, A, 0, 1) == A - 100 * MB);               // commit off only
    CHECK(lp_shortfall(UINT64_MAX, UINT64_MAX, A, C, 1) == 0);               // no overflow

    // episodes: give back at the start, on a deeper shortfall, and at the latest after the retry time
    LpEpisode e = {0};
    CHECK(!lp_episode_step(&e, 0, 0));
    CHECK(lp_episode_step(&e, 100 * MB, 1000));                              // start
    CHECK(!lp_episode_step(&e, 100 * MB, 2000));                             // same
    CHECK(!lp_episode_step(&e, 300 * MB, 3000));   // deeper, but not by the margin (200 < 256)
    CHECK(lp_episode_step(&e, 600 * MB, 4000));                              // grew past the margin from 300
    CHECK(!lp_episode_step(&e, 500 * MB, 5000));                             // shallower
    CHECK(!lp_episode_step(&e, 500 * MB, 4000 + LP_RETRY_MS - 1));
    CHECK(lp_episode_step(&e, 500 * MB, 4000 + LP_RETRY_MS));                // the retry time
    CHECK(!lp_episode_step(&e, 0, 99000));                                   // over
    CHECK(!e.active);
    CHECK(lp_episode_step(&e, 10 * MB, 100000));                             // a new episode starts again

    // sampling: the first call is due, then once a second
    uint64_t next = 0;
    CHECK(lp_sample_due(5000, &next));
    CHECK(!lp_sample_due(5999, &next));
    CHECK(lp_sample_due(6000, &next));
    CHECK(next == 7000);

    if (!failures) printf("linux pressure: all checks passed\n");
    return failures ? 1 : 0;
}
