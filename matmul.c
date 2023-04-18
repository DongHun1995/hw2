#define _GNU_SOURCE
#include "util.h"
#include <immintrin.h>
#include <pthread.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void matmul(const float *A, const float *B, float *C, int M, int N, int K,
            int num_threads)
{

#pragma omp parallel num_threads(num_threads)
  {
#pragma omp for schedule(guided) nowait
    for (int i = 0; i < M; ++i)
    {
      for (int k = 0; k < K; ++k)
      {
        for (int j = 0; j < N; ++j)
        {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
  }
}