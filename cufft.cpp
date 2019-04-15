#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define MIN_LFFT      64ULL
#define MAX_LFFT      (1ULL*1024*1024)
#define MIN_BATCH     128ULL
#define MAX_SAMPLES   (MAX_LFFT*MIN_BATCH)
#define MAX_BATCH     (MAX_SAMPLES/MIN_LFFT)
#define MAX_CPLX_OUT  ((MIN_LFFT*MAX_BATCH)/2 + MAX_BATCH)

int main(int argc, char **argv)
{
  cufftHandle fftplan_r2c;
  cudaDeviceProp dp;
  cudaEvent_t tstart, tstop;
  void *d_in, *h_in, *d_out;
  int device = 0;

  // Select device and do some preparations
  if ((argc == 2) && isdigit(argv[1][0])) {
    device = argv[1][0] - '0';
  }

  checkCudaErrors( cudaGetDeviceProperties(&dp, device) );
  printf("CUDA Device #%d : %s, Compute Capability %d.%d, %d threads/block, warpsize %d\n",
      device, dp.name, dp.major, dp.minor, dp.maxThreadsPerBlock, dp.warpSize
      );

  checkCudaErrors( cudaSetDevice(device) );
  checkCudaErrors( cudaDeviceReset() );
  checkCudaErrors( cudaEventCreate(&tstart) );
  checkCudaErrors( cudaEventCreate(&tstop) );

  checkCudaErrors( cudaMalloc( (void **)&d_in, sizeof(cufftReal)*MAX_SAMPLES ) );
  checkCudaErrors( cudaMalloc( (void **)&d_out, sizeof(cufftComplex)*MAX_CPLX_OUT ) );
  checkCudaErrors( cudaHostAlloc( (void **)&h_in, sizeof(cufftReal)*MAX_SAMPLES, cudaHostAllocDefault ) );
  for (size_t n=0; n<MAX_SAMPLES; n++) {
    ((cufftReal*)h_in)[n] = cufftReal(n % 1234);
  }

  printf("    Nreals     Lbatch T_r2c[ms]  R[Gs/s]\n");

  for (size_t fftlen = MIN_LFFT; fftlen <= MAX_LFFT; fftlen *= 2) {

    size_t batch = MAX_SAMPLES / fftlen;

    // CuFFT R2C plan : N reals = N/2 complex
    int dimn[1] = {fftlen};         // DFT size
    int inembed[1] = {0};           // ignored for 1D xform
    int onembed[1] = {0};           // ignored for 1D xform
    int istride = 1, ostride = 1;   // step between in(out) samples
    int idist = fftlen;             // in step between FFTs (R2C input = real)
    int odist = fftlen/2 + 1;       // out step between FFTs (x2C output = complex); use N/2+0 to discard Nyquist
    checkCudaErrors( cufftPlanMany(&fftplan_r2c, 1, dimn,
	  inembed, istride, idist,
	  onembed, ostride, odist,
	  CUFFT_R2C, batch)
	);

#if defined(CUDA_VERSION) && (CUDA_VERSION < 8000)
    checkCudaErrors( cufftSetCompatibilityMode(fftplan_r2c, CUFFT_COMPATIBILITY_NATIVE) );
#endif

    // Execute FFT once to force instantiation of the plan (if delayed instantiation)
    checkCudaErrors( cufftExecR2C(fftplan_r2c, (cufftReal*)d_in, (cufftComplex*)d_out) );


    // Restore FFT input and output areas to a known state
    checkCudaErrors( cudaMemcpy( d_in, h_in, idist*sizeof(cufftReal)*batch, cudaMemcpyHostToDevice) );
    checkCudaErrors( cudaMemset( d_out, 0x00, odist*sizeof(cufftComplex)*batch ) );

    // Time the execution of cufftExecR2C()
    checkCudaErrors( cudaEventRecord(tstart) );
    checkCudaErrors( cufftExecR2C(fftplan_r2c, (cufftReal*)d_in, (cufftComplex*)d_out) );
    checkCudaErrors( cudaEventRecord(tstop) );
    checkCudaErrors( cudaEventSynchronize(tstop) );

    float dt_msec = 0.0f;
    checkCudaErrors( cudaEventElapsedTime( &dt_msec, tstart, tstop ) );
    printf("%10zu %10zu %9.3f %8.1f\n", fftlen, batch, dt_msec, (1e-6*fftlen*batch)/dt_msec);

    // Cleanup
    checkCudaErrors( cufftDestroy(fftplan_r2c) );
  }

  return 0;
}
