#ifndef __FFT_CUDA_H__
#define __FFT_CUDA_H__

#include <fft.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>

#define THREADS_PER_BLOCK 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

__device__ double2 cuda_complex_sub(double2 a, double2 b);
__device__ double2 cuda_complex_add(double2 a, double2 b);
__device__ double2 cuda_complex_mult(double2 a, double2 b);

__global__ void fft_cuda_reverse(double2 *ip, double2 *op, int m, int size);

extern double2 *dev_ip, *dev_op;

#endif //__FFT_CUDA_H__
