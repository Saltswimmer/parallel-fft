#include "libfft.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

static uint8_t reverse_byte[256] = {
    0b00000000, 0b10000000, 0b01000000, 0b11000000, 0b00100000, 0b10100000,
    0b01100000, 0b11100000, 0b00010000, 0b10010000, 0b01010000, 0b11010000,
    0b00110000, 0b10110000, 0b01110000, 0b11110000, 0b00001000, 0b10001000,
    0b01001000, 0b11001000, 0b00101000, 0b10101000, 0b01101000, 0b11101000,
    0b00011000, 0b10011000, 0b01011000, 0b11011000, 0b00111000, 0b10111000,
    0b01111000, 0b11111000, 0b00000100, 0b10000100, 0b01000100, 0b11000100,
    0b00100100, 0b10100100, 0b01100100, 0b11100100, 0b00010100, 0b10010100,
    0b01010100, 0b11010100, 0b00110100, 0b10110100, 0b01110100, 0b11110100,
    0b00001100, 0b10001100, 0b01001100, 0b11001100, 0b00101100, 0b10101100,
    0b01101100, 0b11101100, 0b00011100, 0b10011100, 0b01011100, 0b11011100,
    0b00111100, 0b10111100, 0b01111100, 0b11111100, 0b00000010, 0b10000010,
    0b01000010, 0b11000010, 0b00100010, 0b10100010, 0b01100010, 0b11100010,
    0b00010010, 0b10010010, 0b01010010, 0b11010010, 0b00110010, 0b10110010,
    0b01110010, 0b11110010, 0b00001010, 0b10001010, 0b01001010, 0b11001010,
    0b00101010, 0b10101010, 0b01101010, 0b11101010, 0b00011010, 0b10011010,
    0b01011010, 0b11011010, 0b00111010, 0b10111010, 0b01111010, 0b11111010,
    0b00000110, 0b10000110, 0b01000110, 0b11000110, 0b00100110, 0b10100110,
    0b01100110, 0b11100110, 0b00010110, 0b10010110, 0b01010110, 0b11010110,
    0b00110110, 0b10110110, 0b01110110, 0b11110110, 0b00001110, 0b10001110,
    0b01001110, 0b11001110, 0b00101110, 0b10101110, 0b01101110, 0b11101110,
    0b00011110, 0b10011110, 0b01011110, 0b11011110, 0b00111110, 0b10111110,
    0b01111110, 0b11111110, 0b00000001, 0b10000001, 0b01000001, 0b11000001,
    0b00100001, 0b10100001, 0b01100001, 0b11100001, 0b00010001, 0b10010001,
    0b01010001, 0b11010001, 0b00110001, 0b10110001, 0b01110001, 0b11110001,
    0b00001001, 0b10001001, 0b01001001, 0b11001001, 0b00101001, 0b10101001,
    0b01101001, 0b11101001, 0b00011001, 0b10011001, 0b01011001, 0b11011001,
    0b00111001, 0b10111001, 0b01111001, 0b11111001, 0b00000101, 0b10000101,
    0b01000101, 0b11000101, 0b00100101, 0b10100101, 0b01100101, 0b11100101,
    0b00010101, 0b10010101, 0b01010101, 0b11010101, 0b00110101, 0b10110101,
    0b01110101, 0b11110101, 0b00001101, 0b10001101, 0b01001101, 0b11001101,
    0b00101101, 0b10101101, 0b01101101, 0b11101101, 0b00011101, 0b10011101,
    0b01011101, 0b11011101, 0b00111101, 0b10111101, 0b01111101, 0b11111101,
    0b00000011, 0b10000011, 0b01000011, 0b11000011, 0b00100011, 0b10100011,
    0b01100011, 0b11100011, 0b00010011, 0b10010011, 0b01010011, 0b11010011,
    0b00110011, 0b10110011, 0b01110011, 0b11110011, 0b00001011, 0b10001011,
    0b01001011, 0b11001011, 0b00101011, 0b10101011, 0b01101011, 0b11101011,
    0b00011011, 0b10011011, 0b01011011, 0b11011011, 0b00111011, 0b10111011,
    0b01111011, 0b11111011, 0b00000111, 0b10000111, 0b01000111, 0b11000111,
    0b00100111, 0b10100111, 0b01100111, 0b11100111, 0b00010111, 0b10010111,
    0b01010111, 0b11010111, 0b00110111, 0b10110111, 0b01110111, 0b11110111,
    0b00001111, 0b10001111, 0b01001111, 0b11001111, 0b00101111, 0b10101111,
    0b01101111, 0b11101111, 0b00011111, 0b10011111, 0b01011111, 0b11011111,
    0b00111111, 0b10111111, 0b01111111, 0b11111111};

static const float PI = (float)acos(-1);

float *fft(float *data, int length, int thread_count, float *output) {
    float complex *buf =
        (float complex *)malloc(length * sizeof(float complex));

    float complex *coefficients =
        (float complex *)malloc(length / 2 * sizeof(float complex));

    // phase 1: pre-compute coefficients
    printf("Phase 1\n");
#pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < length / 2; i++) {
        coefficients[i] = cexpf((i / length) * -2 * PI * I);
    }

    // phase 2: rearrange indices
    // each index gets mapped to the index if you reverse the bits

    printf("Phase 2\n");

    unsigned int bit_size = (unsigned int)ceil(log2((float)length));

    for (unsigned int i = 0; i < (unsigned int)length; i++) {
        uint8_t *i_p = (uint8_t *)&i;
        uint8_t  scratch[sizeof(unsigned int)];

        for (unsigned int j = 0u; j < sizeof(unsigned int); j++) {
            scratch[sizeof(unsigned int) - j - 1] = reverse_byte[i_p[j]];
        }

        unsigned int new_index = ((unsigned int *)scratch)[i];
        new_index = new_index >> (8 * sizeof(unsigned int) - bit_size);
        buf[i] = (float complex)(data[new_index]);
    }

    // phase 3: intra-thread complex number arithmetic

    printf("Phase 3\n");
    /*
    #pragma omp parallel num_threads(thread_count)
        {
    */
    for (int N = 2; N <= length; N *= 2) {
        int coeff_step = length / N;
        for (int i = 0; i < length; i += N) {
            for (int j = i, k = 0; j < (i + N / 2); j++, k += coeff_step) {
                float complex p = buf[j + N / 2] * coefficients[k];
                buf[j + N / 2] = buf[j] - p;
                buf[j] += p;
            }
        }
        if (N == length)
            break;
    }
    //   }

    /*
    // phase 3: inter-thread complex number arithmetic
    printf("Phase 3\n");

    if (thread_count > 1) {
        // clang-format off
        #pragma omp parallel num_threads(thread_count)
        {
            int thread_id = omp_get_thread_num();
            int thread_divisor = 2;

            while (1) {
                if (thread_id % thread_divisor) {
                    break;
                }
                int N = thread_divisor * block_size;
                int coeff_step = N / length;

                for (int k = 0, coeff_idx = 0; k < N / 2; k++, coeff_idx +=
    coeff_step) { float complex p = buf[thread_id * block_size + k]; float
    complex q = buf[thread_id * block_size + k + N / 2] *
                                      coefficients[coeff_idx];
                                      //cexpf(thread_id * -2 * PI * I / N);
                    buf[thread_id * block_size + k] = p + q;
                    buf[thread_id * block_size + k + N / 2] = p - q;
                }

                thread_divisor *= 2;
            }
        }
    }

    free(coefficients);

    // clang-format on
    */
    // phase 4: convert complex numbers to real
    printf("Phase 4\n");

#pragma omp parallel for num_threads(thread_count)
    for (int i = 0; i < length; i++) {
        output[i] = creal(buf[i]);
    }

    free(buf);
    return output;
}
