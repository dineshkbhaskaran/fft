#include <fft_cuda.cuh>

__device__ int reverse(int n, int m)
{
  int j = n;

  j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;
  j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;
  j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;
  j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;

  j >>= (16 - m);

  return j;
}

__global__ void fft_cuda3_kernel(double2 *ip, double2 *op, int m, int size)
{
  __shared__ double2 shared_op[2048];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (4 * tid < size) {
    double2 eps = (double2){0.0, -1.0};

    double2 ip0 = ip[reverse(4*tid+0, m)];
    double2 ip1 = ip[reverse(4*tid+1, m)];
    double2 ip2 = ip[reverse(4*tid+2, m)];
    double2 ip3 = ip[reverse(4*tid+3, m)];

    double2 t0 = cuda_complex_add(ip0, ip1);
    double2 t1 = cuda_complex_add(ip2, ip3);

    double2 s0 = cuda_complex_sub(ip0, ip1);
    double2 s1 = cuda_complex_mult(cuda_complex_sub(ip2, ip3), eps);

    shared_op[4*tid+0] = cuda_complex_add(t0, t1);
    shared_op[4*tid+2] = cuda_complex_sub(t0, t1);

    shared_op[4*tid+1] = cuda_complex_add(s0, s1);
    shared_op[4*tid+3] = cuda_complex_sub(s0, s1);
  }

  __syncthreads();

  for (int i = 2; i < m; i++) {
    int len = 1 << i;  /* the length of half bfly at level m*/
    double2 factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    int bfly_len = (len << 1);
    int nbfly = size / bfly_len;

    if (tid < nbfly) {
      int j = tid * bfly_len;
      double2 omega = {1, 0};

      for (int k = j; k < j+len; k++) {
        double2 temp = cuda_complex_mult(omega, shared_op[k+len]);

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
  double2 *ip = (double2 *)_ip;
  double2 *op = (double2 *)_op;

  gpuErrchk(cudaMemcpy(dev_ip, ip, size*sizeof(double2), cudaMemcpyHostToDevice));
 
  /* Can only work until size 2048 */
  int threads = (512 < size) ? 512 : size;
  dim3 block(threads, 1, 1);
  dim3 grid(size/threads, 1, 1);

  fft_cuda3_kernel<<<grid, block>>> (dev_ip, dev_op, m, size);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(op, dev_op, size*sizeof(double2), cudaMemcpyDeviceToHost));
}
