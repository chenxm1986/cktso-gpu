CKTSO-GPU: GPU-based Sparse Solver Accelerator for Circuit Simulation
============

CKTSO-GPU is a **GPU acceleration** module of [CKTSO](https://github.com/chenxm1986/cktso), a high-performance parallel sparse direct solver specially designed for SPICE-based circuit simulation. CKTSO-GPU provides acceleration for LU re-factorization and forward/backward substitutions for slightly-dense circuit matrices, by using CUDA.

CKTSO-GPU supports both real and complex matrices. Both row and column modes are supported.

Please read "ug.pdf" for more information about the usage of this package. Read "howto.txt" to see how to compile and run the demos.

Notes on Library and Integer Bitwidths
============
Only x86-64 libraries are provided. This means that, a 64-bit Linux operating system is needed.

Functions for both 32-bit integers and 64-bit integers are provided. The latter has '_L' in the function names. The integer bitwidth only limits the size of the input matrix. The internal data structures always use 64-bit integers.

Author
============
Please visit [Xiaoming Chen's personal page](http://people.ucas.edu.cn/~chenxm).
