#include <fft_cuda.cuh>

__global__ void fft_cuda2_kernel(complex_t *ip, complex_t *op, int m, int size)
{
  __shared__ complex_t shared_op[2048];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (tid == 0) {
    for (int i = 0, j = 0, k; i < size-1; i++) {
      if (i <= j) {
        shared_op[j] = ip[i];
        shared_op[i] = ip[j];
      }

      for (k = size/2; k <= j; k >>= 1) {
        j -= k;
      }
      j += k;
    }

    shared_op[size-1] = ip[size-1];
  }

  __syncthreads();

  for (int i = 0; i < m; i++) {
    int len = 1 << i;  /* the length of half block at level m*/
    complex_t factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    int block_len = (len << 1);
    int nblocks = size / block_len;

    if (tid < nblocks) {
      int j = tid * block_len;
      complex_t omega = {1, 0};

      for (int k = j; k < j+len; k++) {
        complex_t temp = cuda_complex_mult(omega, shared_op[k+len]);

        shared_op[k+len] = cuda_complex_sub(shared_op[k], temp);
        shared_op[k    ] = cuda_complex_add(shared_op[k], temp);

        omega = cuda_complex_mult(omega, factor);
      }
    }

    __syncthreads();
  }

  for (; tid < size; tid += blockDim.x) {
    op[tid] = shared_op[tid];
  }
}

void fft_cuda2(complex_t *_ip, complex_t *_op, int size)
{
  int m = (int)log2((double)size);
  complex_t *ip = (complex_t *)_ip;
  complex_t *op = (complex_t *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(complex_t), cudaMemcpyHostToDevice));
 
  /* Can only work until size 2048 */
  int threads = (1024 < size) ? 1024 : size;
  dim3 block(threads, 1, 1);
  dim3 grid(size/threads, 1, 1);

  fft_cuda2_kernel<<<grid, block>>> (dev_ip, dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(complex_t), cudaMemcpyDeviceToHost));
}
