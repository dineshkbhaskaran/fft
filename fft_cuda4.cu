#include <fft_cuda.cuh>

__global__ void fft_cuda4_kernel(complex_t *ip, complex_t *op, int m, int size)
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
    int bfly_size = (1 << (i + 1));
    int bfly_half = (1 << i);

    int k = tid & ((1 << i ) - 1);
    int bfly_idx = tid >> i;

    if (tid >= size/2) {
      continue;
    }
    
    double angle = (-2.0 * PI * k / bfly_size);
    complex_t omega = {cos(angle), sin(angle)};

    int off = bfly_idx * bfly_size;
    complex_t temp = cuda_complex_mult(omega, shared_op[off + k + bfly_half]);

    shared_op[off + k + bfly_half] = cuda_complex_sub(shared_op[off + k], temp);
    shared_op[off + k]             = cuda_complex_add(shared_op[off + k], temp);

    __syncthreads();
  }


  if (2 * tid < size) {
    op[2*tid+0] = shared_op[2*tid+0];
    op[2*tid+1] = shared_op[2*tid+1];
  }
}

void fft_cuda4(complex_t *_ip, complex_t *_op, int size)
{
  int m = (int)log2((double)size);
  complex_t *ip = (complex_t *)_ip;
  complex_t *op = (complex_t *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(complex_t), cudaMemcpyHostToDevice));
 
  /* Can only work until size 2048 */
  int threads = (128 < size) ? 128 : size;
  dim3 block(threads, 1, 1);
  dim3 grid(size/threads, 1, 1);

  fft_cuda4_kernel<<<grid, block>>> (dev_ip, dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(complex_t), cudaMemcpyDeviceToHost));
}
