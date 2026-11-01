"""Fused CUDA ant construction: one device thread constructs one complete ant.

The SplitMix64 stream is private to each ant/iteration. No cuRAND library or
CPU-to-GPU transfer of random draws is needed. Optional CuPy is imported only
when this module's constructor is called.
"""

from functools import lru_cache

import numpy as np

_SOURCE = r"""
extern "C" __device__ unsigned long long next_random(unsigned long long *state) {
    unsigned long long z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void construct(
    const double *dist, const double *times, const double *work,
    const double *demand, const double *log_weights,
    unsigned char *visited, double *weights, long long *routes,
    double *lengths, long long *dive_counts,
    const int n, const int ants, const double endurance,
    const double capacity, const double tolerance,
    const unsigned long long seed, const unsigned long long iteration
) {
    int ant = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant >= ants) return;
    const long long width = n + 1;
    unsigned char *seen = visited + ant * width;
    double *scores = weights + ant * width;
    long long *route = routes + ant * (2LL * n + 1);
    for (int j = 0; j <= n; ++j) seen[j] = 0;
    seen[0] = 1;
    int current = 0, remaining = n, cursor = 0, dives = 0;
    double elapsed = 0.0, collected = 0.0, total = 0.0;
    unsigned long long state = seed ^ ((ant + 1ULL) * 0xd2b74407b1ce6e93ULL)
                                    ^ ((iteration + 1ULL) * 0xca5a826395121157ULL);
    while ((remaining > 0 || current != 0) && cursor < 2 * n) {
        double maximum = -1.0e300;
        bool any = false;
        for (int j = 0; j <= n; ++j) {
            bool allowed = j == 0 ? current != 0 :
                (!seen[j] && remaining > 0 &&
                 elapsed + times[current * width + j] + work[j] + times[j * width]
                     <= endurance + tolerance && collected + demand[j] <= capacity);
            scores[j] = allowed ? log_weights[current * width + j] : -1.0e300;
            if (allowed) {
                any = true;
                maximum = scores[j] > maximum ? scores[j] : maximum;
            }
        }
        if (!any) { lengths[ant] = 1.0/0.0; dive_counts[ant] = 1; return; }
        double sum = 0.0;
        for (int j = 0; j <= n; ++j) {
            scores[j] = scores[j] <= -1.0e300 ? 0.0 : exp(scores[j] - maximum);
            sum += scores[j];
        }
        double uniform = (next_random(&state) >> 11) * (1.0 / 9007199254740992.0);
        double threshold = uniform * sum, cumulative = 0.0;
        int selected = 0;
        for (int j = 0; j <= n; ++j) {
            if (scores[j] > 0.0) selected = j; // last-positive roundoff fallback
            cumulative += scores[j];
            if (threshold < cumulative) { selected = j; break; }
        }
        total += dist[current * width + selected];
        route[++cursor] = selected;
        if (selected == 0) { ++dives; elapsed = 0.0; collected = 0.0; }
        else {
            elapsed += times[current * width + selected] + work[selected];
            collected += demand[selected];
            seen[selected] = 1;
            --remaining;
        }
        current = selected;
    }
    lengths[ant] = remaining == 0 && current == 0 ? total : 1.0/0.0;
    dive_counts[ant] = dives;
}
"""


@lru_cache(maxsize=1)
def _kernel():
    import cupy

    return cupy.RawKernel(_SOURCE, "construct", options=("--std=c++11",))


def construct_colony(
    xp,
    distances,
    times,
    work,
    demand,
    log_weights,
    *,
    ants,
    endurance,
    capacity,
    tolerance,
    seed,
    iteration,
):
    n = len(work) - 1
    visited = xp.empty((ants, n + 1), dtype=xp.uint8)
    scratch = xp.empty((ants, n + 1), dtype=xp.float64)
    routes = xp.zeros((ants, 2 * n + 1), dtype=xp.int64)
    lengths = xp.empty(ants, dtype=xp.float64)
    dives = xp.empty(ants, dtype=xp.int64)
    _kernel()(
        ((ants + 127) // 128,),
        (128,),
        (
            distances,
            times,
            work,
            demand,
            log_weights,
            visited,
            scratch,
            routes,
            lengths,
            dives,
            np.int32(n),
            np.int32(ants),
            np.float64(endurance),
            np.float64(capacity),
            np.float64(tolerance),
            np.uint64(seed % (2**64)),
            np.uint64(iteration),
        ),
    )
    return routes, lengths, dives
