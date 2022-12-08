/*CKTSO-GPU is a GPU acceleration module of CKTSO*/
/*Version 20221207*/
#ifndef __CKTSO_GPU__
#define __CKTSO_GPU__

#include "cktso.h"

/********** error code **********
* >0 for CUDA runtime error code (refer to cudaError_t). some common errors:
* cudaErrorMemoryAllocation = 2
* cudaErrorInsufficientDriver = 35
* cudaErrorCallRequiresNewerDriver = 36
* cudaErrorInvalidDeviceFunction = 98
* cudaErrorNoDevice = 100
* cudaErrorNoKernelImageForDevice = 209
* Besides CKTSO error codes, CKTSO-GPU also uses following additional error codes:
* -51:  insufficient GPU resources
* -52:  GPU-accelerator not initialized
* -53:  matrix not refactorized by GPU
********************************/

/********** input parameters int [] **********
* input parm[0]: timer. [default 0]: no timer | >0: microsecond/us-level timer | <0: millisecond/ms-level timer
* input parm[1]: threshold for bulk-pipeline refactor. [default 128]
* input parm[2]: factors array allocation ratio (percentage). [default 110 (=1.1)]
* input parm[3]: whether to reallocate memories when size reduces. [default 0]
* input parm[4]: #blocks per multiprocessor, for refactor. [default 0] automatic decision
* input parm[5]: #threads per block, for refactor. [default 0] automatic decision
* input parm[6]: #blocks per multiprocessor, for solve. [default 0] automatic decision
* input parm[7]: #threads per block, for solve. [default 0] automatic decision
********************************/

/********** output parameters const long long [] **********
* output parm[0]: time (in microsecond/us) of CKTSO(_L)_InitializeGpuAccelerator
* output parm[1]: time (in microsecond/us) of CKTSO(_L)_GpuRefactorize
* output parm[2]: time (in microsecond/us) of CKTSO(_L)_GpuSolve
* output parm[3]: host memory usage (in bytes)
* output parm[4]: GPU memory usage (in bytes)
* output parm[5]: host memory requirement (in bytes) when -4 is returned (for the last malloc/realloc failure)
* output parm[6]: GPU memory requirement (in bytes) when cudaErrorMemoryAllocation (2) is returned (for the last cudaMalloc failure)
********************************/

typedef struct __CKTSO_GPU *ICktSoGpu;
typedef struct __CKTSO_L_GPU *ICktSoGpu_L;

#ifdef __cplusplus
extern "C" {
#endif

/*
* CKTSO_CreateGpuAccelerator (CKTSO_L_CreateGpuAccelerator): creates GPU-accelerator instance and retrieves parameter array pointers
* @accel: pointer to an ICktSoGpu (ICktSoGpu_L) instance that retrieves created GPU-accelerator instance handle
* @iparm: pointer to input parameter list array (see annotations above)
* @oparm: pointer to output parameter list array (see annotations above)
* @gpuid: GPU id for refactor and solve
*/
int CKTSO_CreateGpuAccelerator
(
	_OUT_ ICktSoGpu *accel, 
	_OUT_ int **iparm, 
	_OUT_ const long long **oparm, 
	_IN_ int gpuid
);

int CKTSO_L_CreateGpuAccelerator
(
	_OUT_ ICktSoGpu_L *accel,
	_OUT_ int **iparm,
	_OUT_ const long long **oparm,
	_IN_ int gpuid
);

/*
* CKTSO_InitializeGpuAccelerator (CKTSO_L_InitializeGpuAccelerator): initializes GPU-accelerator data
* Each time when LU factors structure has been changed (i.e., CKTSO_Factorize (CKTSO_L_Factorize) has been called), GPU-accelerator data needs to be re-initialized
* CKTSO_SortFactors (CKTSO_L_SortFactors) is recommended before calling this routine
* @accel: GPU-accelerator instance handle returned by CKTSO_CreateGpuAccelerator (CKTSO_L_CreateGpuAccelerator)
* @inst: solver instance handle returned by CKTSO_CreateSolver (CKTSO_L_CreateSolver)
*/
int CKTSO_InitializeGpuAccelerator
(
	_IN_ ICktSoGpu accel, 
	_IN_ ICktSo inst
);

int CKTSO_L_InitializeGpuAccelerator
(
	_IN_ ICktSoGpu_L accel,
	_IN_ ICktSo_L inst
);

/*
* CKTSO_DestroyGpuAccelerator (CKTSO_L_DestroyGpuAccelerator): destroys GPU-accelerator
* @accel: GPU-accelerator instance handle returned by CKTSO_CreateGpuAccelerator (CKTSO_L_CreateGpuAccelerator)
*/
int CKTSO_DestroyGpuAccelerator
(
	_IN_ ICktSoGpu accel
);

int CKTSO_L_DestroyGpuAccelerator
(
	_IN_ ICktSoGpu_L accel
);

/*
* CKTSO_GpuRefactorize (CKTSO_L_GpuRefactorize): refactorizes matrix without partial pivoting
* Call this routine after CKTSO_InitializeGpuAccelerator (CKTSO_L_InitializeGpuAccelerator) has been called
* @accel: GPU-accelerator instance handle returned by CKTSO_CreateGpuAccelerator (CKTSO_L_CreateGpuAccelerator)
* @ax: double array of length ap[n], matrix values, in host memory
*/
int CKTSO_GpuRefactorize
(
	_IN_ ICktSoGpu accel,
	_IN_ const double ax[]
);

int CKTSO_L_GpuRefactorize
(
	_IN_ ICktSoGpu_L accel,
	_IN_ const double ax[]
);

/*
* CKTSO_GpuSolve (CKTSO_L_GpuSolve): solves solution
* Call this routine after CKTSO_GpuRefactorize (CKTSO_L_GpuRefactorize) has been called
* @b: double array of length n to specify right-hand-side vector, in host memory
* @x: double array of length n to get solution, in host memory
* @row0_column1: row or column mode
*/
int CKTSO_GpuSolve
(
	_IN_ ICktSoGpu accel,
	_IN_ const double b[],
	_OUT_ double x[], /*x address can be same as b address*/
	_IN_ bool row0_column1
);

int CKTSO_L_GpuSolve
(
	_IN_ ICktSoGpu_L accel,
	_IN_ const double b[],
	_OUT_ double x[], /*x address can be same as b address*/
	_IN_ bool row0_column1
);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
struct __CKTSO_GPU
{
	virtual int _CDECL_ DestroyGpuAccelerator
	(
	) = 0;
	virtual int _CDECL_ InitializeGpuAccelerator
	(
		_IN_ ICktSo inst
	) = 0;
	virtual int _CDECL_ GpuRefactorize
	(
		_IN_ const double ax[]
	) = 0;
	virtual int _CDECL_ GpuSolve
	(
		_IN_ const double b[],
		_OUT_ double x[], /*x address can be same as b address*/
		_IN_ bool row0_column1
	) = 0;
};

struct __CKTSO_L_GPU
{
	virtual int _CDECL_ DestroyGpuAccelerator
	(
	) = 0;
	virtual int _CDECL_ InitializeGpuAccelerator
	(
		_IN_ ICktSo_L inst
	) = 0;
	virtual int _CDECL_ GpuRefactorize
	(
		_IN_ const double ax[]
	) = 0;
	virtual int _CDECL_ GpuSolve
	(
		_IN_ const double b[],
		_OUT_ double x[], /*x address can be same as b address*/
		_IN_ bool row0_column1
	) = 0;
};
#endif

#endif
