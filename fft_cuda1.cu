#include <fft_cuda.cuh>

__global__ void fft_cuda1_kernel(complex_t *op, int m, int size)
{
  for (int i = 0; i < m; i++) {
    int len = 1 << i;  /* the length of half array */
    complex_t factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    for (int j = 0; j < size; j += 2*len) {
      complex_t omega = {1, 0};

      for (int k = j; k < j+len; k++) {
        complex_t temp = cuda_complex_mult(omega, op[k+len]);

        op[k+len] = cuda_complex_sub(op[k], temp);
        op[k    ] = cuda_complex_add(op[k], temp);

        omega = cuda_complex_mult(omega, factor);
      }
    }
  }
}

void fft_cuda1(complex_t *_ip, complex_t *_op, int size)
{
  int m = (int)log2((double)size);
  complex_t *ip = (complex_t *)_ip;
  complex_t *op = (complex_t *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(complex_t), cudaMemcpyHostToDevice));
 
  int threads = (THREADS_PER_BLOCK < size) ? THREADS_PER_BLOCK : size;
  dim3 block(threads, 1, 1);

  dim3 grid(size/threads, 1, 1);

  fft_cuda_reverse<<<grid, block>>>(dev_ip, dev_op, m, size); 
  gpuErrchk(cudaPeekAtLastError());
  fft_cuda1_kernel<<<1, 1>>> (dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(complex_t), cudaMemcpyDeviceToHost));
}
