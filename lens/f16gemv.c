/* lens/f16gemv.c - OUR OWN fused float16 GEMV for the decode path (PLAN item (f2b)).
 *
 * WHY THIS FILE EXISTS
 * (f2) measured, on both attention ships, that int8/f16 weights are 4.0000x / 2.0000x fewer bytes per
 * decoded token and that EVERY numpy form of using them is SLOWER (1.689x to 4.741x the float32
 * reference): numpy has no fast low-precision GEMV, and every "widen, then multiply" form writes a
 * 4-byte-wide copy of the weights and reads it back, so the saving is spent on the widening
 * (lens/runs/INT8-DECODE.txt, micro-probes in lens/runs/_tmp/). The byte win is collectable ONLY by a
 * kernel that widens INSIDE the multiply and never materialises the float32 copy. This is that kernel.
 *
 * WHAT IT COMPUTES (exactly, and nothing else):
 *     y[r, j] = sum_i x[r, i] * W[i, j]        W is row-major float16 (IEEE 754 binary16 payload)
 * No bias, no activation: the caller adds those, so the only thing that differs from the numpy arms is
 * where the widening happens.
 *
 * LOOP ORDER IS THE WHOLE POINT: for each input row i, one sequential sweep over that row's f16
 * payload, accumulating into the output vector. W is read once, in order, at 2 bytes per weight, and
 * the accumulated y stays in L1/L2. The transposed order (output-major) would touch a different cache
 * line per weight and waste 32x the bandwidth - that mistake is not made here.
 *
 * THREADS: m == 1 (the decode case) is split over the INPUT dimension with one accumulator per thread
 * and a final reduction, because a single row of answer.W is 12002 outputs and a per-row split would
 * leave most of the machine idle. m > 1 (seeding the window, m = W) is split over the ROWS instead, so
 * no two threads ever write the same output element.
 *
 * NO EXTERNAL MATH LIBRARY. No BLAS, no framework. gcc's auto-vectoriser plus OpenMP, exactly like
 * lens/csrc/kern.c. The prior on this machine is that our own hand-written C lost 3.1x to OpenBLAS at
 * large COMPUTE-bound GEMMs; this is a memory-bound GEMV whose premise is fewer bytes read, which is
 * the one regime that loss does not cover. It is measured, not argued.
 *
 * Build (the loader in lens/fusedgemv.py does this automatically):
 *     gcc -O3 -march=native -fopenmp -funroll-loops -shared -fPIC -o f16gemv.so lens/f16gemv.c
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define F16GEMV_MAXTHREADS 64

/* ------------------------------------------------------------------ f16 -> f32
 * The conversion must be EXACT: binary16 values are all representable in binary32. Two independent
 * paths are provided so they can be checked against each other and against numpy:
 *   f16_to_f32        - the fast path (the F16C instruction when the compiler has it)
 *   f16_to_f32_sw     - the software path, always compiled, exported for the test
 */
#if defined(__FLT16_MANT_DIG__)
typedef _Float16 f16_t;
static inline float f16_to_f32(uint16_t h) {
    f16_t v;
    memcpy(&v, &h, sizeof v);
    return (float)v;
}
#else
static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;                                  /* +/- 0 */
        } else {
            int shift = 0;                             /* subnormal: normalise the mantissa */
            while ((m & 0x400u) == 0) { m <<= 1; shift++; }
            m &= 0x3FFu;
            bits = s | ((uint32_t)(127 - 14 - shift) << 23) | (m << 13);
        }
    } else if (e == 31u) {
        bits = s | 0x7F800000u | (m << 13);            /* inf / nan */
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);     /* 127 - 15 = 112 */
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}
#endif

/* The software path, exported: the test checks it against numpy AND against the fast path. */
void f16_decode_sw(const uint16_t *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t h = in[i];
        uint32_t s = (h & 0x8000u) << 16;
        uint32_t e = (h >> 10) & 0x1Fu;
        uint32_t m = h & 0x3FFu;
        uint32_t bits;
        if (e == 0) {
            if (m == 0) {
                bits = s;
            } else {
                int shift = 0;
                while ((m & 0x400u) == 0) { m <<= 1; shift++; }
                m &= 0x3FFu;
                bits = s | ((uint32_t)(127 - 14 - shift) << 23) | (m << 13);
            }
        } else if (e == 31u) {
            bits = s | 0x7F800000u | (m << 13);
        } else {
            bits = s | ((e + 112u) << 23) | (m << 13);
        }
        float f;
        memcpy(&f, &bits, sizeof f);
        out[i] = f;
    }
}

/* The fast path, exported for the same check. */
void f16_decode_fast(const uint16_t *in, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(in[i]);
}

/* ------------------------------------------------------------------ per-thread accumulators */
static float *g_acc[F16GEMV_MAXTHREADS];
static size_t g_acc_n[F16GEMV_MAXTHREADS];

static int ensure_acc(int t, size_t n_out) {
    if (t < 0 || t >= F16GEMV_MAXTHREADS) return -1;
    if (g_acc[t] != NULL && g_acc_n[t] >= n_out) return 0;
    free(g_acc[t]);
    g_acc[t] = (float *)malloc(n_out * sizeof(float));
    if (g_acc[t] == NULL) { g_acc_n[t] = 0; return -1; }
    g_acc_n[t] = n_out;
    return 0;
}

/* ------------------------------------------------------------------ the kernel */
/* returns 0 on success, negative on a bad argument or an allocation failure. */
int f16_gemv_f32(const uint16_t *W, const float *x, float *y,
                 int m, int n_in, int n_out, int nthreads) {
    if (W == NULL || x == NULL || y == NULL) return -1;
    if (m <= 0 || n_in <= 0 || n_out <= 0) return -1;

    int nt = nthreads;
    if (nt < 1) nt = 1;
#ifdef _OPENMP
    {
        int mx = omp_get_max_threads();
        if (nt > mx) nt = mx;
    }
#else
    nt = 1;
#endif
    if (nt > F16GEMV_MAXTHREADS) nt = F16GEMV_MAXTHREADS;

    /* ---- m > 1: one thread per (row, input) pair, sequential sweep over that row's inputs.
     * The output is ZEROED FIRST in its own pass: with a flat parallel index a row's elements can be
     * split across two threads, so an "initialise on i == 0" trick would race and is not used. ---- */
    if (m > 1 || nt == 1) {
        long long total = (long long)m * (long long)n_in;
        long long nout = (long long)m * (long long)n_out;
        for (long long k = 0; k < nout; k++) y[k] = 0.0f;
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
        for (long long idx = 0; idx < total; idx++) {
            int r = (int)(idx / n_in);
            int i = (int)(idx % n_in);
            const float xi = x[(size_t)r * n_in + i];
            const uint16_t *wr = W + (size_t)i * n_out;
            float *yr = y + (size_t)r * n_out;
#pragma omp simd
            for (int j = 0; j < n_out; j++) yr[j] += xi * f16_to_f32(wr[j]);
        }
        return 0;
    }

    /* ---- m == 1, threaded: split the INPUT dimension, one accumulator per thread ---- */
    for (int t = 0; t < nt; t++) {
        if (ensure_acc(t, (size_t)n_out) != 0) return -2;
    }
#pragma omp parallel num_threads(nt)
    {
        int t = 0;
#ifdef _OPENMP
        t = omp_get_thread_num();
#endif
        int i0 = (int)((long long)n_in * t / nt);
        int i1 = (int)((long long)n_in * (t + 1) / nt);
        float *acc = g_acc[t];
        for (int j = 0; j < n_out; j++) acc[j] = 0.0f;
        for (int i = i0; i < i1; i++) {
            const float xi = x[i];
            const uint16_t *wr = W + (size_t)i * n_out;
#pragma omp simd
            for (int j = 0; j < n_out; j++) acc[j] += xi * f16_to_f32(wr[j]);
        }
    }
    for (int j = 0; j < n_out; j++) {
        float s = 0.0f;
        for (int t = 0; t < nt; t++) s += g_acc[t][j];
        y[j] = s;
    }
    return 0;
}

/* A deliberately slow reference: same formula, no vectoriser, no threads, one output at a time.
 * Exported so the test can show the fast path is not merely "the same code". */
int f16_gemv_f32_scalar(const uint16_t *W, const float *x, float *y,
                        int m, int n_in, int n_out) {
    if (W == NULL || x == NULL || y == NULL || m <= 0 || n_in <= 0 || n_out <= 0) return -1;
    for (int r = 0; r < m; r++) {
        for (int j = 0; j < n_out; j++) {
            float s = 0.0f;
            for (int i = 0; i < n_in; i++) {
                s += x[(size_t)r * n_in + i] * f16_to_f32(W[(size_t)i * n_out + j]);
            }
            y[(size_t)r * n_out + j] = s;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ the AVX2 / F16C kernel
 * WHY A SECOND KERNEL: the first version above (i-major, `#pragma omp simd`) was MEASURED at 4.5-9 GB/s
 * against numpy's float32 GEMV at 52-131 GB/s on the same shapes in the same process
 * (lens/runs/F16-GEMV-BENCH.txt). That gap is far too large to be the byte count - it means the loop
 * was not being compiled into good code - so before any conclusion is drawn about whether the byte win
 * is collectable, the kernel has to be written the way a GEMV has to be written:
 *
 *   * ONE BLOCK OF OUTPUTS AT A TIME, its accumulators held IN YMM REGISTERS across the whole input
 *     loop. The version above loads and stores the accumulator from memory once per weight, so it
 *     moves ~10 bytes per element where 2 bytes of weight are the point.
 *   * `_mm256_cvtph_ps` widens 8 float16 to 8 float32 in ONE instruction, so the float32 copy of the
 *     weights is never written - the whole reason numpy lost on the f16 arm.
 *   * The j-blocks are the PARALLEL unit (not the input rows): a block writes its own outputs and no
 *     other block touches them, so there is no reduction and no per-thread buffer. This matters because
 *     the decode case is m == 1, where splitting over input rows would leave 11 of 12 threads idle.
 *   * Block-outer/input-inner is also the cache-correct order here: a 16-element block of answer.W
 *     touches 41 rows x 32 bytes = 41 cache lines (2.6 KB, L1-resident), and the next block touches the
 *     SAME lines at the other half, so the weights are read exactly once.
 */
#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>

/* One 16-wide output block of one row: y[j..j+15] = sum_i x[i] * W[i, j..j+15]. */
static inline void f16_block16(const uint16_t *W, const float *xr, float *yr,
                               int n_in, int n_out, int j) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    for (int i = 0; i < n_in; i++) {
        const uint16_t *wr = W + (size_t)i * (size_t)n_out + j;
        const __m256 vx = _mm256_set1_ps(xr[i]);
        a0 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wr)), a0);
        a1 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wr + 8))), a1);
    }
    _mm256_storeu_ps(yr + j, a0);
    _mm256_storeu_ps(yr + j + 8, a1);
}

/* One 8-wide block (the tail when n_out is not a multiple of 16). */
static inline void f16_block8(const uint16_t *W, const float *xr, float *yr,
                              int n_in, int n_out, int j) {
    __m256 a0 = _mm256_setzero_ps();
    for (int i = 0; i < n_in; i++) {
        const uint16_t *wr = W + (size_t)i * (size_t)n_out + j;
        const __m256 vx = _mm256_set1_ps(xr[i]);
        a0 = _mm256_fmadd_ps(vx, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)wr)), a0);
    }
    _mm256_storeu_ps(yr + j, a0);
}

int f16_gemv_f32_avx2(const uint16_t *W, const float *x, float *y,
                      int m, int n_in, int n_out, int nthreads) {
    if (W == NULL || x == NULL || y == NULL) return -1;
    if (m <= 0 || n_in <= 0 || n_out <= 0) return -1;
    int nt = nthreads;
    if (nt < 1) nt = 1;
#ifdef _OPENMP
    {
        int mx = omp_get_max_threads();
        if (nt > mx) nt = mx;
    }
#else
    nt = 1;
#endif
    if (nt > F16GEMV_MAXTHREADS) nt = F16GEMV_MAXTHREADS;

    const int nblk = (n_out + 15) / 16;
    const long long total = (long long)m * (long long)nblk;
#pragma omp parallel for schedule(static) num_threads(nt) if (nt > 1)
    for (long long idx = 0; idx < total; idx++) {
        const int r = (int)(idx / nblk);
        const int j = (int)(idx % nblk) * 16;
        const float *xr = x + (size_t)r * (size_t)n_in;
        float *yr = y + (size_t)r * (size_t)n_out;
        if (j + 16 <= n_out) {
            f16_block16(W, xr, yr, n_in, n_out, j);
        } else if (j + 8 <= n_out) {
            f16_block8(W, xr, yr, n_in, n_out, j);
            for (int k = j + 8; k < n_out; k++) {
                float s = 0.0f;
                for (int i = 0; i < n_in; i++) s += xr[i] * f16_to_f32(W[(size_t)i * n_out + k]);
                yr[k] = s;
            }
        } else {
            for (int k = j; k < n_out; k++) {
                float s = 0.0f;
                for (int i = 0; i < n_in; i++) s += xr[i] * f16_to_f32(W[(size_t)i * n_out + k]);
                yr[k] = s;
            }
        }
    }
    return 0;
}
#endif
