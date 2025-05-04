#include "libfft.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

// clang-format off
void partial_fft(
    float *data,
    int stride,
    int N,
    float complex *buf) 
{
    // clang-format on
    if (N == 1) {
        buf[0] = (float complex)(data[0]);
        return;
    }

    partial_fft(data, stride * 2, N / 2, buf);
    partial_fft(data + stride, stride * 2, N / 2, buf + N / 2);
#pragma omp parallel for
    for (int i = 0; i < N / 2; i++) {
        float complex p = buf[i];

        const float   PI = (float)acos(-1);
        float complex q = buf[i + N / 2] * cexpf(i * -2 * PI * I / N);
        buf[i] = p + q;
        buf[i + N / 2] = p - q;
    }
}

float *fft(float *data, int length, int thread_count, float *output) {
    float complex *buf =
        (float complex *)malloc(length * sizeof(float complex));

    // This algorithm can only realize a performance benefit on up to length /2
    // threads
    if (length / 2 < thread_count) {
        thread_count = length / 2;
    }

    // clang-format off

    //#pragma omp parallel num_threads(thread_count)
    {
        partial_fft(data, 1, length, buf);

        #pragma omp parallel for
        for (int i = 0; i < length; i++) {
            output[i] = creal(buf[i]);
        }
    }
    // clang-format on

    free(buf);
    return output;
}
