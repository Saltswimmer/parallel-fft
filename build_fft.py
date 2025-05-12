'''
Name: Ethan Ciavolella
Id: 916328107
Homework: Final Project
See README.md for instructions on how to compile and run the program
'''
from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef("""
    void fft(float *, int, int, float *);
""")
ffibuilder.set_source("_libfft_cffi",
"""
    #include "libfft.h"
""",
    sources=['libfft.c'],
    extra_compile_args=['-fopenmp', '-O3', '-Wall'],
    extra_link_args=['-fopenmp'])

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
