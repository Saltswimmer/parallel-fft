#include "libfft.h"
#include <stdio.h>
#include <omp.h>

void        hello() {
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < 10; i++) {
        printf("Hello, world %d!\n", i);
    }
}
