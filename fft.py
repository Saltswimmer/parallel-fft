from _libfft_cffi import ffi, lib
from scipy.io import wavfile
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import wave
import timeit

def libfft_demo(data, padded_length, num_bins, num_threads):

        c_data = ffi.new("float[]", data.tolist())
        output = ffi.new("float[]", [0.0 for i in range(padded_length)])
        lib.fft(c_data, padded_length, num_threads, output)

        return [i for i in output[0:num_bins]]

def main():
    parser = argparse.ArgumentParser(
        prog='parallel-fft',
        description='Parallel implementation of fast Fourier transform algorithm'
    )
    parser.add_argument('filename')
    parser.add_argument('num_threads', type=int)
    parser.add_argument('-s', '--scipy', action='store_true')

    args = parser.parse_args()

    samplerate, data = wavfile.read(args.filename)

    duration = data.shape[0] / samplerate
    time = np.linspace(0., duration, data.shape[0])

    # Pad the sample so that the total number of samples is a power
    # of two. This simplifies the algorithm.
    padded_length = int(math.exp2(math.ceil(math.log2(data.shape[0]))))
    pad_data = np.pad(data, (0, padded_length - data.shape[0]), 'constant', constant_values=(0, 0))

    num_bins = math.ceil(2000 * padded_length / samplerate)
    freq = np.linspace(0., 2000., num_bins)

    result_time = 0.0
    num_trials = 5
    if args.scipy:
         result_time = timeit.timeit(lambda: fft.fft(pad_data), number=num_trials) 
    
         print(f"Average run time: {result_time / num_trials} seconds")

         fig = plt.figure(layout='constrained')
         plots = fig.subplots(1, 2, squeeze=False)

         output = fft.fft(pad_data)

         plots[0, 0].set_xlabel('Time (seconds)')
         plots[0, 0].set_ylabel('Amplitude')
         plots[0, 0].plot(time, data)
 
         plots[0, 1].set_xlabel('Frequency (hz)')
         plots[0, 1].set_ylabel('Intensity')
         plots[0, 1].plot(freq, output[0:num_bins])
         plot = fft.fft(data, samplerate)
         plt.show() 
    else:
         result_time = timeit.timeit(lambda: libfft_demo(pad_data, padded_length, num_bins, args.num_threads), number=num_trials) 
         print(f"Average run time: {result_time / num_trials} seconds")

         fig = plt.figure(layout='constrained')
         plots = fig.subplots(1, 2, squeeze=False)

         duration = data.shape[0] / samplerate
         time = np.linspace(0., duration, data.shape[0])

         output = libfft_demo(pad_data, padded_length, num_bins, args.num_threads)

         plots[0, 0].set_xlabel('Time (seconds)')
         plots[0, 0].set_ylabel('Amplitude')
         plots[0, 0].plot(time, data)

         plots[0, 1].set_xlabel('Frequency (hz)')
         plots[0, 1].set_ylabel('Intensity')
         plots[0, 1].plot(freq, output)
         plt.show() 

if __name__ == "__main__":
    main()
