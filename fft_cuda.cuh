#ifndef __FFT_CUDA_H__
#define __FFT_CUDA_H__

#include <fft.h>

#define THREADS_PER_BLOCK 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__device__ complex_t cuda_complex_sub(complex_t a, complex_t b);
__device__ complex_t cuda_complex_add(complex_t a, complex_t b);
__device__ complex_t cuda_complex_mult(complex_t a, complex_t b);

__device__ int reverse(int j, int m);
__global__ void fft_cuda_reverse(complex_t *ip, complex_t *op, int m, int size);
__global__ void fft_cuda_bfly4(complex_t *ip, complex_t *op, int m, int size);

extern complex_t *dev_ip, *dev_op;

#endif //__FFT_CUDA_H__
