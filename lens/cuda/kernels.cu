/* lens/cuda/kernels.cu — our own CUDA kernels, compiled at runtime by NVRTC.
 *
 * Nothing here comes from a library. Every kernel is a plain __global__ function written for
 * this project. Compiled to PTX in-process and loaded with cuModuleLoadData.
 *
 * Target: GTX 1660 SUPER, Turing sm_75, 22 SMs, 6 GB.
 */

extern "C" {

/* ---------------------------------------------------------------- elementwise */
__global__ void k_add(const float *a, const float *b, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = a[i] + b[i];
}

__global__ void k_mul(const float *a, const float *b, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = a[i] * b[i];
}

__global__ void k_relu(const float *x, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

__global__ void k_sigmoid(const float *x, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void k_mul_scalar(const float *x, float s, float *out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = x[i] * s;
}

/* ---------------------------------------------------------------- tiled matmul
 * C[m,n] = A[m,k] @ B[k,n], row-major.
 *
 * The CPU version is a blocked loop nest and loses to OpenBLAS. This is the reason the GPU
 * layer exists: a tiled matmul on 22 SMs with shared memory reuse. Each block computes a
 * TS x TS tile; both A and B tiles are staged in shared memory once and reused TS times.
 *
 * TS = 16 keeps shared memory at 2 * 16 * 16 * 4 = 2 KB per block, so occupancy is limited by
 * registers rather than shared memory on this card.
 */
#define TS 16

__global__ void k_matmul(const float *A, const float *B, float *C, int m, int k, int n) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (k + TS - 1) / TS; ++t) {
        int aCol = t * TS + threadIdx.x;
        int bRow = t * TS + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[(long)row * k + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? B[(long)bRow * n + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) C[(long)row * n + col] = acc;
}

/* C[m,n] = A[m,k] @ B[n,k]^T -- the dA gradient.
 *
 * BUG FIXED HERE: the B tile was loaded with `col` (which depends on threadIdx.x) where it
 * needed a row index that depends on threadIdx.y. Every thread then read the wrong row of B and
 * the result was unrelated to the correct answer -- the ops gate measured rel diff ~1.4, i.e.
 * nothing matched. The tile row must be blockIdx.x*TS + threadIdx.y.
 */
__global__ void k_matmul_abt(const float *A, const float *B, float *C, int m, int k, int n) {
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];

    int row = blockIdx.y * TS + threadIdx.y;       // row of A and of C
    int col = blockIdx.x * TS + threadIdx.x;       // column of C
    int bRow = blockIdx.x * TS + threadIdx.y;      // row of B -- THE FIX
    float acc = 0.0f;

    for (int t = 0; t < (k + TS - 1) / TS; ++t) {
        int aCol = t * TS + threadIdx.x;
        int bCol = t * TS + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? A[(long)row * k + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && bCol < k) ? B[(long)bRow * k + bCol] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TS; ++i) acc += As[threadIdx.y][i] * Bs[threadIdx.x][i];
        __syncthreads();
    }
    if (row < m && col < n) C[(long)row * n + col] = acc;
}

/* ---------------------------------------------------------------- fused softmax + CE
 * One block per row. A 1501-way row fits in shared memory at 1501 * 4 = 6 KB.
 * MAXV is the compile-time cap; rows longer than this are rejected in Python, not silently
 * truncated. Raising it costs shared memory per block.
 */
#define MAXV 4096

__global__ void k_softmax_ce(const float *logits, const int *labels, int rows, int V,
                             float *probs, float *dlogits) {
    int r = blockIdx.x;
    if (r >= rows) return;
    extern __shared__ float row[];
    const float *src = logits + (long)r * V;

    int t = threadIdx.x;
    float mx = -1e30f;
    for (int j = t; j < V; j += blockDim.x) {
        row[j] = src[j];
        if (src[j] > mx) mx = src[j];
    }
    /* block max */
    __shared__ float red[MAXV > 1024 ? 1024 : MAXV];
    red[t] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s && red[t + s] > red[t]) red[t] = red[t + s];
        __syncthreads();
    }
    mx = red[0];
    __syncthreads();

    float s = 0.0f;
    for (int j = t; j < V; j += blockDim.x) s += expf(row[j] - mx);
    red[t] = s;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (t < step) red[t] += red[t + step];
        __syncthreads();
    }
    s = red[0];
    __syncthreads();

    float inv = 1.0f / s;
    int lab = labels[r];
    for (int j = t; j < V; j += blockDim.x) {
        float p = expf(row[j] - mx) * inv;
        if (probs) probs[(long)r * V + j] = p;
        if (dlogits) {
            dlogits[(long)r * V + j] = p - (j == lab ? 1.0f : 0.0f);
        }
    }
}

} /* extern "C" */
