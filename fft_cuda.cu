#include <fft_cuda.cuh>

#define TWO_PI (-6.28318530717958647692)

complex_t *dev_ip, *dev_op, *dev_bm;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %d %s %d\n", cudaGetErrorString(code), code, file, line);
    if (abort) {
      exit(code);
    }
  }
}

void cuda_init()
{
  complex_t bm[4096];

  for (int i = 0; i < 4096; i++) {
    bm[i] = (complex_t){cos(TWO_PI*i/4096), sin(TWO_PI*i/4096)};
  }

  gpuErrchk(cudaMalloc((void **)&dev_ip, 4096*sizeof(complex_t))); 
  gpuErrchk(cudaMalloc((void **)&dev_op, 4096*sizeof(complex_t))); 
  gpuErrchk(cudaMalloc((void **)&dev_bm, 4096*sizeof(complex_t))); 

  gpuErrchk(cudaMemcpy(dev_bm, bm, 4096*sizeof(complex_t), cudaMemcpyHostToDevice));
}

void cuda_free()
{
  gpuErrchk(cudaFree(dev_ip));
  gpuErrchk(cudaFree(dev_op));
}

#define THREADS_PER_BLOCK 128

__device__ int reverse(int j, int m)
{
  j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;
  j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;
  j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;
  j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;

  j >>= (16 - m);

  return j;
}

__global__ void fft_cuda_reverse(complex_t *ip, complex_t *op, int m, int size)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  unsigned int j = tid;

  j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;
  j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;
  j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;
  j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;

  j >>= (16 - m);

  if (tid > size) {
    return;
  }

  op[j] = ip[tid];
  op[tid] = ip[j];
}

__global__ void fft_cuda_bfly4(complex_t *ip, complex_t *op, int m, int size)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (4 * tid < size) {
    complex_t eps = (complex_t){0.0, -1.0};

    complex_t ip0 = ip[reverse(4*tid+0, m)];
    complex_t ip1 = ip[reverse(4*tid+1, m)];
    complex_t ip2 = ip[reverse(4*tid+2, m)];
    complex_t ip3 = ip[reverse(4*tid+3, m)];

    complex_t t0 = cuda_complex_add(ip0, ip1);
    complex_t t1 = cuda_complex_add(ip2, ip3);

    complex_t s0 = cuda_complex_sub(ip0, ip1);
    complex_t s1 = cuda_complex_mult(cuda_complex_sub(ip2, ip3), eps);

    op[4*tid+0] = cuda_complex_add(t0, t1);
    op[4*tid+1] = cuda_complex_add(s0, s1);

    op[4*tid+2] = cuda_complex_sub(t0, t1);
    op[4*tid+3] = cuda_complex_sub(s0, s1);
  }
}

__global__ void fft_cuda_bfly8(complex_t *ip, complex_t *op, int m, int size)
{
  int tid = (threadIdx.x + blockDim.x * blockIdx.x) * 8;

  if (tid < size) {
    complex_t eps = (complex_t){0.0, -1.0};

    complex_t ip0 = ip[reverse(tid+0, m)];
    complex_t ip1 = ip[reverse(tid+1, m)];
    complex_t ip2 = ip[reverse(tid+2, m)];
    complex_t ip3 = ip[reverse(tid+3, m)];
    complex_t ip4 = ip[reverse(tid+4, m)];
    complex_t ip5 = ip[reverse(tid+5, m)];
    complex_t ip6 = ip[reverse(tid+6, m)];
    complex_t ip7 = ip[reverse(tid+7, m)];

    complex_t t0 = cuda_complex_add(ip0, ip1);
    complex_t t1 = cuda_complex_add(ip2, ip3);
    complex_t t2 = cuda_complex_add(ip4, ip5);
    complex_t t3 = cuda_complex_add(ip6, ip7);

    complex_t s0 = cuda_complex_sub(ip0, ip1);
    complex_t s1 = cuda_complex_mult(cuda_complex_sub(ip2, ip3), eps);
    complex_t s2 = cuda_complex_sub(ip4, ip5);
    complex_t s3 = cuda_complex_mult(cuda_complex_sub(ip6, ip7), eps);

    complex_t p0 = cuda_complex_add(t0, t1);
    complex_t p1 = cuda_complex_add(s0, s1);
    complex_t p2 = cuda_complex_sub(t0, t1);
    complex_t p3 = cuda_complex_sub(s0, s1);

    complex_t p4 = cuda_complex_add(t2, t3);
    complex_t p5 = cuda_complex_add(s2, s3);
    complex_t p6 = cuda_complex_sub(t2, t3);
    complex_t p7 = cuda_complex_sub(s2, s3);

    complex_t eps0 = (complex_t) {1.0, 0.0};
    //complex_t eps1 = (complex_t) {cos(TWO_PI/8), sin(TWO_PI/8)};
    complex_t eps1 = (complex_t) {0.70710678118654752440084436210485, -0.70710678118654752440084436210485};
    complex_t eps2 = (complex_t) {0.0, -1.0};
    //complex_t eps3 = (complex_t) {-cos(TWO_PI/8), sin(TWO_PI/8)};
    complex_t eps3 = (complex_t) {-0.70710678118654752440084436210485, -0.70710678118654752440084436210485};

    op[tid+0] = cuda_complex_add(p0, cuda_complex_mult(p4, eps0));
    op[tid+1] = cuda_complex_add(p1, cuda_complex_mult(p5, eps1));
    op[tid+2] = cuda_complex_add(p2, cuda_complex_mult(p6, eps2));
    op[tid+3] = cuda_complex_add(p3, cuda_complex_mult(p7, eps3));
             
    op[tid+4] = cuda_complex_sub(p0, cuda_complex_mult(p4, eps0));
    op[tid+5] = cuda_complex_sub(p1, cuda_complex_mult(p5, eps1));
    op[tid+6] = cuda_complex_sub(p2, cuda_complex_mult(p6, eps2));
    op[tid+7] = cuda_complex_sub(p3, cuda_complex_mult(p7, eps3));
  }
}

__device__ complex_t cuda_complex_sub(complex_t a, complex_t b)
{
  return (complex_t) {a.x - b.x, a.y - b.y};
}

__device__ complex_t cuda_complex_add(complex_t a, complex_t b)
{
  return (complex_t) {a.x + b.x, a.y + b.y};
}

__device__ complex_t cuda_complex_mult(complex_t a, complex_t b)
{
  return (complex_t) {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
