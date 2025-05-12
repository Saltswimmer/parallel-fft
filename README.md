# parallel-fft
Parallel implementation of fast Fourier transform using OpenMP

By Ethan Ciavolella

For CS04391 Parallel and Concurrent Programming

## Installation

(Optional but recommended) Set up a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:
```
python -m pip install -r requirements.txt
```

Compile the library:
```
python build_fft.py
```

Run the program:
```
python fft.py <filename> <number of threads> (-s)
```
## Credits

Reference implementation of iterative serial FFT: https://www.nayuki.io/page/free-small-fft-in-multiple-languages

Online tone generator: https://www.szynalski.com/tone-generator/

speleothem.wav sample file provided by Ethan Ciavolella
