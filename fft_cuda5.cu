#include <fft_cuda.cuh>

__global__ void fft_cuda5_kernel(complex_t *ip, complex_t *op, int m, int size)
{
  /* We handle two elements per thread */
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  int bfly_size = (1 << (m + 1));
  int bfly_half = (1 << m);

  int k = tid & ((1 << m ) - 1);
  int bfly_idx = tid >> m;

  if (tid >= size/2) {
    return;
  }

  double angle = (-2.0 * PI * k / bfly_size);
  complex_t omega = {cos(angle), sin(angle)};

  int off = bfly_idx * bfly_size + k;

  complex_t temp1 = op[off];
  complex_t temp2 = op[off + bfly_half];
  
  complex_t temp = cuda_complex_mult(omega, temp2);

  op[off]             = cuda_complex_add(temp1, temp);
  op[off + bfly_half] = cuda_complex_sub(temp1, temp);
}

void fft_cuda5(complex_t *_ip, complex_t *_op, int size)
{
  int m = (int)log2((double)size);
  complex_t *ip = (complex_t *)_ip;
  complex_t *op = (complex_t *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(complex_t), cudaMemcpyHostToDevice));
 

  int threads = (128 < size/4) ? 128 : size/4;
  dim3 block1(threads, 1, 1);
  dim3 grid1(size/(threads*4), 1, 1);

  fft_cuda_bfly4<<<grid1, block1>>>(dev_ip, dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  threads = 128;
  threads = (threads < size) ? threads : size;
  dim3 block2(threads, 1, 1);
  dim3 grid2(size/threads, 1, 1);

  for (int i = 2; i < m; i++) {
    fft_cuda5_kernel<<<grid2, block2>>> (dev_ip, dev_op, i, size);
    gpuErrchk(cudaPeekAtLastError());
  }

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(complex_t), cudaMemcpyDeviceToHost));
}
