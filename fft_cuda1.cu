#include <fft_cuda.cuh>

__global__ void fft_cuda1_kernel(double2 *op, int m, int size)
{
  for (int i = 0; i < m; i++) {
    int len = 1 << i;  /* the length of half array */
    double2 factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    for (int j = 0; j < size; j += 2*len) {
      double2 omega = {1, 0};

      for (int k = j; k < j+len; k++) {
        double2 temp = cuda_complex_mult(omega, op[k+len]);

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
  double2 *ip = (double2 *)_ip;
  double2 *op = (double2 *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(double2), cudaMemcpyHostToDevice));
 
  int threads = (THREADS_PER_BLOCK < size) ? THREADS_PER_BLOCK : size;
  dim3 block(threads, 1, 1);

  dim3 grid(size/threads, 1, 1);

  fft_cuda_reverse<<<grid, block>>>(dev_ip, dev_op, m, size); 
  gpuErrchk(cudaPeekAtLastError());
  fft_cuda1_kernel<<<1, 1>>> (dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(double2), cudaMemcpyDeviceToHost));
}
