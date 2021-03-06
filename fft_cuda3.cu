#include <fft_cuda.cuh>

__global__ void fft_cuda3_kernel(complex_t *ip, complex_t *op, int m, int size)
{
  __shared__ complex_t shared_op[2048];
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

    shared_op[4*tid+0] = cuda_complex_add(t0, t1);
    shared_op[4*tid+2] = cuda_complex_sub(t0, t1);

    shared_op[4*tid+1] = cuda_complex_add(s0, s1);
    shared_op[4*tid+3] = cuda_complex_sub(s0, s1);
  }

  __syncthreads();

  for (int i = 2; i < m; i++) {
    int len = 1 << i;  /* the length of half bfly at level m*/
    complex_t factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    int bfly_len = (len << 1);
    int nbfly = size / bfly_len;

    if (tid < nbfly) {
      int j = tid * bfly_len;
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

  __syncthreads();

  if (2 * tid < size) {
    op[2*tid+0] = shared_op[2*tid+0];
    op[2*tid+1] = shared_op[2*tid+1];
  }
}

void fft_cuda3(complex_t *_ip, complex_t *_op, int size)
{
  int m = (int)log2((double)size);
  complex_t *ip = (complex_t *)_ip;
  complex_t *op = (complex_t *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(complex_t), cudaMemcpyHostToDevice));
 
  /* Can only work until size 2048 */
  int threads = (128 < size) ? 128 : size;
  dim3 block(threads, 1, 1);
  dim3 grid(size/threads, 1, 1);

  fft_cuda3_kernel<<<grid, block>>> (dev_ip, dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(complex_t), cudaMemcpyDeviceToHost));
}
