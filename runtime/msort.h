// A stable merge sort with the comparison inlined, for the per-submission range lists of vk_memory.c and vk_usnap.c.
// A profile of the hills region in Bananza 3.2.0 had about 3 ms a frame on the submitting thread in the CRT's qsort
// under copy_ranges and wr_prune: a call through a pointer for every comparison, over lists of thousands of ranges
// that are mostly in order already. This one returns at once for a sorted list, sorts blocks of 32 by insertion and
// merges them bottom-up through a buffer the caller keeps.
//
// MSORT_DEFINE(name, T, LESS) defines static void name(T *a, size_t n, T **buf, size_t *cap), where LESS(x, y) is an
// expression on two const T * that is nonzero when *x sorts before *y. *buf grows with realloc and stays the caller's.
#pragma once
#include <stdlib.h>
#include <string.h>

#define MSORT_DEFINE(name, T, LESS)                                                                                    \
    static void name(T *a, size_t n, T **buf, size_t *cap) {                                                          \
        size_t s = 1;                                                                                                  \
        while (s < n && !(LESS((&a[s]), (&a[s - 1])))) s++;                                                            \
        if (s >= n) return;                                                                                            \
        if (*cap < n) {                                                                                                \
            T *nb = (T *)realloc(*buf, n * sizeof(T));                                                                 \
            if (!nb) abort();                                                                                          \
            *buf = nb, *cap = n;                                                                                       \
        }                                                                                                              \
        for (size_t lo = 0; lo < n; lo += 32) {                                                                        \
            size_t hi = lo + 32 < n ? lo + 32 : n;                                                                     \
            for (size_t i = lo + 1; i < hi; i++) {                                                                     \
                T v = a[i];                                                                                            \
                size_t j = i;                                                                                          \
                while (j > lo && LESS((&v), (&a[j - 1]))) a[j] = a[j - 1], j--;                                        \
                a[j] = v;                                                                                              \
            }                                                                                                          \
        }                                                                                                              \
        T *src = a, *dst = *buf;                                                                                       \
        for (size_t w = 32; w < n; w *= 2) {                                                                           \
            for (size_t lo = 0; lo < n; lo += 2 * w) {                                                                 \
                size_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, i = lo, j = mid, k = lo;   \
                if (mid == hi || !(LESS((&src[mid]), (&src[mid - 1])))) {                                              \
                    memcpy(dst + lo, src + lo, (hi - lo) * sizeof(T));   /* already in order */                       \
                    continue;                                                                                          \
                }                                                                                                      \
                while (i < mid && j < hi) dst[k++] = LESS((&src[j]), (&src[i])) ? src[j++] : src[i++];                 \
                while (i < mid) dst[k++] = src[i++];                                                                   \
                while (j < hi) dst[k++] = src[j++];                                                                    \
            }                                                                                                          \
            T *t = src;                                                                                                \
            src = dst, dst = t;                                                                                        \
        }                                                                                                              \
        if (src != a) memcpy(a, src, n * sizeof(T));                                                                   \
    }
