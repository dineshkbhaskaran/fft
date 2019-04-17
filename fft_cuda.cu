#include <fft_cuda.cuh>

complex_t *dev_ip;
complex_t *dev_op;

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
  gpuErrchk(cudaMalloc((void **)&dev_ip, 4096*sizeof(complex_t))); 
  gpuErrchk(cudaMalloc((void **)&dev_op, 4096*sizeof(complex_t))); 
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
    op[4*tid+2] = cuda_complex_sub(t0, t1);

    op[4*tid+1] = cuda_complex_add(s0, s1);
    op[4*tid+3] = cuda_complex_sub(s0, s1);
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
