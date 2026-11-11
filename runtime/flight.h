// TOTK_FLIGHT=1 flight recorder (flight.c): the last frames' timing and event counts plus a thumbnail each, written to
// disk on F8 or after a slow frame, so a glitch or stutter seen in play can be read after the fact.
#pragma once
#include <stdint.h>

#define FLIGHT_THUMB_W 320
#define FLIGHT_THUMB_H 180

extern int flight_on;   // TOTK_FLIGHT=1, set once by flight_init before the first present

void flight_init(void);   // prof_start calls it
// perf_present, once per present on the render thread: the interval since the previous present, the render thread's
// blocked time per PERF_* kind in this frame, and the running PERF_CNT_* totals.
void flight_frame(uint64_t frame, uint64_t now_ns, uint64_t interval_ns, const uint64_t blocked_ns[],
                  const volatile int64_t counts[]);

// The thumbnails are the graphics back end's (gfx.h GfxBackend.thumbs_enable, thumbs_read): its present keeps every
// presented frame's in a ring of FLIGHT_THUMB_W x FLIGHT_THUMB_H slots.
