/* glasscore.c — our own compute kernels. Written from scratch for GlassNet.
 *
 * No BLAS. No external math library. No framework. This file IS the compute layer.
 * Built with: -O3 -march=native -ffast-math -fopenmp
 *
 * Conventions:
 *   - all arrays are contiguous, row-major, float32
 *   - every function is a plain loop nest; the only help is OpenMP threads and the
 *     compiler's auto-vectoriser
 *   - nothing here allocates: the caller owns every buffer
 *
 * "Fast" honestly means: 100-1000x faster than the same loops in Python, and in the same
 * ballpark as a tuned BLAS for the shapes this project uses. It does not mean faster than
 * OpenBLAS at large sizes, and nothing here claims that.
 */
#include <stddef.h>
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------- elementwise */
/* Memory-bound. Simple loops; the compiler vectorises them and OpenMP splits them. */

void gc_add_f32(const float *a, const float *b, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void gc_sub_f32(const float *a, const float *b, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void gc_mul_f32(const float *a, const float *b, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void gc_div_f32(const float *a, const float *b, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] / b[i];
}

void gc_add_scalar_f32(const float *a, float s, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + s;
}

void gc_mul_scalar_f32(const float *a, float s, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * s;
}

/* ---------------------------------------------------------------- activations */

void gc_relu_f32(const float *x, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void gc_sigmoid_f32(const float *x, float *out, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + expf(-x[i]));
}

/* ---------------------------------------------------------------- reductions */

float gc_sum_f32(const float *x, size_t n) {
    double acc = 0.0;   /* accumulate in double: float sums lose precision fast */
#pragma omp parallel for reduction(+ : acc) schedule(static)
    for (size_t i = 0; i < n; ++i) acc += (double)x[i];
    return (float)acc;
}

/* ---------------------------------------------------------------- matmul
 * C[m,n] = A[m,k] @ B[k,n], row-major.
 *
 * Loop order i-k-j (not i-j-k): the inner loop walks one row of B and one row of C
 * contiguously, so both streams come from cache-friendly addresses. i-j-k would stride
 * through B by n floats per step and thrash.
 *
 * Blocking over k keeps the working set inside L2 for the shapes we use. KC is chosen
 * small enough that a row block of A and a row block of C stay resident.
 */
#define GC_KC 256

void gc_matmul_f32(const float *A, const float *B, float *C, int m, int k, int n) {
    memset(C, 0, (size_t)m * (size_t)n * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        float *crow = C + (size_t)i * (size_t)n;
        for (int kk = 0; kk < k; kk += GC_KC) {
            int kmax = kk + GC_KC < k ? kk + GC_KC : k;
            for (int p = kk; p < kmax; ++p) {
                const float a = A[(size_t)i * (size_t)k + (size_t)p];
                if (a == 0.0f) continue;          /* sparse rows are common here */
                const float *brow = B + (size_t)p * (size_t)n;
                for (int j = 0; j < n; ++j) crow[j] += a * brow[j];
            }
        }
    }
}

/* A^T @ B, used by every backward pass. Transposed reads of A are strided, so the
 * loop order is i-k-j over A^T, which means walking a column of A per output row. */
void gc_matmul_atb_f32(const float *A, const float *B, float *C, int k, int m, int n) {
    /* A is (k,m); C is (m,n) = A^T @ B where B is (k,n) */
    memset(C, 0, (size_t)m * (size_t)n * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        float *crow = C + (size_t)i * (size_t)n;
        for (int p = 0; p < k; ++p) {
            const float a = A[(size_t)p * (size_t)m + (size_t)i];
            if (a == 0.0f) continue;
            const float *brow = B + (size_t)p * (size_t)n;
            for (int j = 0; j < n; ++j) crow[j] += a * brow[j];
        }
    }
}

/* A @ B^T, needed by the backward pass: dA = grad @ b^T.
 * A is (m,k), B is (n,k), C is (m,n).
 * The inner loop is a dot product over two CONTIGUOUS rows, so it vectorises well.
 * (The first version of the backward used gc_matmul_atb for this and the shapes did not
 *  even match -- caught by the L1 gate with "matmul_atb shape mismatch: (12,8)^T @ (16,8)".) */
void gc_matmul_abt_f32(const float *A, const float *B, float *C, int m, int k, int n) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const float *arow = A + (size_t)i * (size_t)k;
        float *crow = C + (size_t)i * (size_t)n;
        for (int j = 0; j < n; ++j) {
            const float *brow = B + (size_t)j * (size_t)k;
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) acc += arow[p] * brow[p];
            crow[j] = acc;
        }
    }
}

/* ---------------------------------------------------------------- fused losses
 * Softmax + cross-entropy in one pass, and it returns the per-row gradient at the same
 * time. Fusing matters: the unfused version writes the probability matrix to memory and
 * reads it back, which for a 1501-way output is the dominant cost.
 *
 * logits: (rows, V) row-major. labels: (rows,) int32.
 * probs_out may be NULL if only the loss is wanted. dlogits_out may be NULL too.
 */
float gc_softmax_ce_f32(const float *logits, const int *labels, int rows, int V,
                        float *probs_out, float *dlogits_out) {
    double total = 0.0;
#pragma omp parallel for reduction(+ : total) schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float *row = logits + (size_t)r * (size_t)V;
        /* max for numerical stability */
        float mx = row[0];
        for (int j = 1; j < V; ++j) if (row[j] > mx) mx = row[j];
        double s = 0.0;
        for (int j = 0; j < V; ++j) s += expf(row[j] - mx);
        const float inv = (float)(1.0 / s);
        /* -log p[label] = -(z_label - mx - log s) = -(z_label - mx) + log s
         * The first version had "- log(s)" here, a sign error. It was caught because the
         * probability matrix and the gradient still matched NumPy to 1e-8 while the loss was
         * off by 5.755 -- so the bug had to be in this one line and nowhere else. */
        total += -(double)(row[labels[r]] - mx) + log(s);
        if (probs_out) {
            float *prow = probs_out + (size_t)r * (size_t)V;
            for (int j = 0; j < V; ++j) prow[j] = expf(row[j] - mx) * inv;
        }
        if (dlogits_out) {
            float *drow = dlogits_out + (size_t)r * (size_t)V;
            for (int j = 0; j < V; ++j) drow[j] = expf(row[j] - mx) * inv;
            drow[labels[r]] -= 1.0f;
        }
    }
    return (float)(total / (double)rows);
}
