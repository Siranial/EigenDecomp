/*
 * EigenDecomposition.cuh
 *
 *  Created on: March 18, 2024
 *      Author: Jonathan Adams
 */

#ifndef EIGENDECOMPOSITION_CUH_
#define EIGENDECOMPOSITION_CUH_

#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_DIM 16

void CUDAEigenDecomp(double*, int, double*, double*, int);
__global__ void CUDASerialNormalize(double*, int);
__global__ void CUDAMatMult(double*, double*, int, double*);
__global__ void CUDASqMatTranspose(double*, int, double*);
__global__ void CUDACopyEigVals(double*, double*, int, int, int);
__global__ void CUDAWPDecrement(double*, double*, int, int);
__global__ void CUDASetDiag(double*, int, double);
__global__ void CUDASetEigVals(double*, int, double*);

//UNUSED FUNCTIONS
__host__ void CUDANormalize(double*, int, int);
__global__ void CUDAVecSqSum(double*, int, double*);
__global__ void CUDADivideByConstant(double*, double, int);


#endif // EIGENDECOMPOSITION_CUH_