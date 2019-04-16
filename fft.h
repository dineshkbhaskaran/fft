#ifndef __FFT_H__
#define __FFT_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_SIZE 4096
#define PI (3.14159265358979323846)

#define diff_time(_x1, _x2) \
    (double)(((_x1.tv_nsec + _x1.tv_sec * 1000000000) - (_x2.tv_nsec + _x2.tv_sec * 1000000000)) / 1000)

struct complex_s {
  /* Real and Imaginary values */
  double r, i;
};

typedef struct complex_s complex_t;

complex_t complex_sub(complex_t a, complex_t b);
complex_t complex_add(complex_t a, complex_t b);
complex_t complex_mult(complex_t a, complex_t b);

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void fft_cuda1(complex_t *_ip, complex_t *_op, int size);
EXTERNC void fft_cuda2(complex_t *_ip, complex_t *_op, int size);
EXTERNC void fft_cuda3(complex_t *_ip, complex_t *_op, int size);
EXTERNC void cuda_init();
EXTERNC void cuda_free();

#endif //__FFT_H__
