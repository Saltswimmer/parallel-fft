from _libfft_cffi import ffi, lib
from scipy.io import wavfile
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import wave
import timeit

def main():
    parser = argparse.ArgumentParser(
        prog='parallel-fft',
        description='Parallel implementation of fast Fourier transform algorithm'
    )
    parser.add_argument('filename')
    parser.add_argument('num_threads', type=int)

    args = parser.parse_args()

    samplerate, data = wavfile.read(args.filename)

    duration = data.shape[0] / samplerate

    # Pad the sample so that the total number of samples is a power
    # of two. This simplifies the algorithm.
    padded_length = int(math.exp2(math.ceil(math.log2(data.shape[0]))))
    pad_data = np.pad(data, (0, padded_length - data.shape[0]), 'constant', constant_values=(0, 0))

    time = np.linspace(0., duration, data.shape[0])
    
    num_bins = math.ceil(2000 * padded_length / samplerate)
    freq = np.linspace(0., 2000., num_bins)

    c_data = ffi.new("float[]", pad_data.tolist())
    output = ffi.new("float[]", [0.0 for i in range(padded_length)])
    my_time = timeit.timeit(lambda: lib.fft(c_data, padded_length, args.num_threads, output), number=1)

    print(f"Completed my fft in {my_time} seconds")

    fig = plt.figure(layout='constrained')
    plots = fig.subplots(1, 2, squeeze=False)

    # temp
    their_time = timeit.timeit(lambda: fft.fft(pad_data, padded_length), number=1)
    print(f"Completed their fft in {my_time} seconds")

    comp_fft = fft.fft(pad_data, padded_length)
    #plots[0, 0].plot(time, data[:])
    plots[0, 0].plot(freq, comp_fft[0:num_bins])
    #plots[0, 0].xlabel("Time (s)")
    #plots[0, 0].ylabel("Amplitude")

    temp_output = [i for i in output[0:num_bins]]
    plots[0, 1].plot(freq, temp_output)

    plt.show() 

if __name__ == "__main__":
    main()
