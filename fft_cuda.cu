#include <fft_cuda.cuh>

double2 *dev_ip;
double2 *dev_op;

void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

void cuda_init()
{
  gpuErrchk(cudaMalloc((void **)&dev_ip, 4096*sizeof(double2))); 
  gpuErrchk(cudaMalloc((void **)&dev_op, 4096*sizeof(double2))); 
}

void cuda_free()
{
  gpuErrchk(cudaFree(dev_ip));
  gpuErrchk(cudaFree(dev_op));
}

#define THREADS_PER_BLOCK 128

__global__ void fft_cuda_reverse(double2 *ip, double2 *op, int m, int size)
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

__device__ double2 cuda_complex_sub(double2 a, double2 b)
{
  return (double2) {a.x - b.x, a.y - b.y};
}

__device__ double2 cuda_complex_add(double2 a, double2 b)
{
  return (double2) {a.x + b.x, a.y + b.y};
}

__device__ double2 cuda_complex_mult(double2 a, double2 b)
{
  return (double2) {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
