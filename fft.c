#include <fft.h>

void print_array(complex_t *a, int n)
{
#ifdef DEBUG
  for (int i = 0; i < n; i++) {
    if (i != 0 && i % 4 == 0) printf ("\n");
    printf ("%10.4f %10.4f ", a[i].x, a[i].y);
  }

  printf ("\n");
#endif 
}

int compare(complex_t *a, complex_t *b, int size)
{
  double epsilon = 10e-6;

  for (int i = 0; i < size; i++) {
    if (fabs(a[i].x - b[i].x) > epsilon) {
      printf ("Arrays dont match at %d %f %f\n", i, a[i].x, b[i].x);
      return 1;
    } else if (fabs(a[i].y - b[i].y) > epsilon) {
      printf ("Arrays dont match at %d %f %f\n", i, a[i].y, b[i].y);
      return 1;
    }
  }

  return 0;
}

complex_t complex_sub(complex_t a, complex_t b)
{
  return (complex_t) {a.x - b.x, a.y - b.y};
}

complex_t complex_add(complex_t a, complex_t b)
{
  return (complex_t) {a.x + b.x, a.y + b.y};
}

complex_t complex_mult(complex_t a, complex_t b)
{
  return (complex_t) {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

/* Recursive FFT */
void fft_recursive(complex_t *ip, complex_t *op, int size)
{
  if (size == 1) {
    op[0] = ip[0];
    return;
  }

  complex_t *ip_even = malloc(sizeof(complex_t) * size/2);
  complex_t *ip_odd  = malloc(sizeof(complex_t) * size/2);

  complex_t *op_even = malloc(sizeof(complex_t) * size/2);
  complex_t *op_odd  = malloc(sizeof(complex_t) * size/2);
  
  for (int i = 0; i < size/2; i++) {
    ip_even[i] = ip[2*i];
    ip_odd[i] = ip[2*i + 1];
  }

  fft_recursive(ip_even, op_even, size/2);
  fft_recursive(ip_odd, op_odd, size/2);

  complex_t factor = {cos(-2.0 * PI / size), sin(-2.0 * PI / size)};
  complex_t omega = {1, 0};
  for (int i = 0; i < size/2; i++) {
    complex_t temp = complex_mult(omega, op_odd[i]);

    op[i       ] = complex_add(op_even[i], temp);
    op[i+size/2] = complex_sub(op_even[i], temp);

    omega = complex_mult(omega, factor);
  }

  free(ip_even);
  free(op_even);
  free(ip_odd);
  free(op_odd);
}

void fft_serial(complex_t *ip, complex_t *op, int size)
{
  int k;
  int m = (int)log2(size);

  /* Load op with reversed bits index values */
  for (int i = 0, j = 0; i < size-1; i++) {
    if (i <= j) {
      op[j] = ip[i];
      op[i] = ip[j];
    }

    for (k = size/2; k <= j; k >>= 1) {
      j -= k;
    }

    j += k;
  }

  op[size-1] = ip[size-1];


  for (int i = 0; i < m; i++) {
    int len = 1 << i;  /* the length of half array */
    complex_t factor = {cos(-2.0 * PI / (2 * len)), sin(-2.0 * PI / (2 * len))};

    for (int j = 0; j < size; j += 2*len) {
      complex_t omega = {1, 0};

      for (int k = j; k < j+len; k++) {
        complex_t temp = complex_mult(omega, op[k+len]);

        op[k+len] = complex_sub(op[k], temp);
        op[k    ] = complex_add(op[k], temp);

        omega = complex_mult(omega, factor);
      }
    }
  }
}

void dft(complex_t *ip, complex_t *op, int size)
{
  for (int k = 0; k < size; k++) {
    complex_t temp = {0.0, 0.0};
    for (int n = 0; n < size; n++) {
      double theta = -2.0 * PI * k * n / size;
      complex_t angle = { cos(theta), sin(theta) };
      temp = complex_add(temp, complex_mult(ip[n], angle));
    }

    op[k] = temp;
  }
}

typedef void (*fft_t)(complex_t *, complex_t *, int);

void run(char *name, complex_t *reference_data, fft_t fft, complex_t *ip, 
    complex_t *op, int size, int clear)
{
  struct timespec stime, etime;

  printf ("\nRunning %s -----------------------------------------\n", name);
  clock_gettime(CLOCK_REALTIME, &stime);
  fft(ip, op, size);
  clock_gettime(CLOCK_REALTIME, &etime);

  if (reference_data != NULL) {
    compare(reference_data, op, size);
  }

  print_array(op, size);
  printf ("\ntime taken %f us\n", diff_time(etime, stime));
  if (clear) {
    memset(op, 0, sizeof(complex_t)*size);
  }
}

int main (int argc, char *argv[])
{
  int size = 1024;
  complex_t ip[MAX_SIZE], op[MAX_SIZE], rop[MAX_SIZE];

  if (argc > 1) { 
    size = atoi(argv[1]);
  }

  memset(ip, 0, sizeof(complex_t)*size);
  memset(op, 0, sizeof(complex_t)*size);
  memset(rop, 0, sizeof(complex_t)*size);

  srand(5126);
  for (int i = 0; i < size; i++) {
    ip[i].x = rand() % 255;
  }

  cuda_init();
  run("dft", NULL, dft, ip, rop, size, 0);
  run("fft-recursive", rop, fft_recursive, ip, op, size, 1);
  run("fft-serial", rop, fft_serial, ip, op, size, 1);
  run("fft-cuda1", rop, fft_cuda1, ip, op, size, 1);
  run("fft-cuda2", rop, fft_cuda2, ip, op, size, 1);
  run("fft-cuda3", rop, fft_cuda3, ip, op, size, 1);
  run("fft-cuda4", rop, fft_cuda4, ip, op, size, 1);
  run("fft-cuda5", rop, fft_cuda5, ip, op, size, 1);

  cuda_free();

  return 0;
}

