/*
 * main.cpp
 *
 *  Created on: March 18, 2024
 *      Author: Jonathan Adams
 *
 *      Implementation of the QR algorithm in CUDA C++ and C++ for computing
 *      eigen vectors and eigen values of a given real symmetric matrix
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cassert>
#include "MatrixOps.hpp"
#include "Memory.hpp"
#include "EigenDecomp.cuh"

int main(int argc, char* argv[]) {
    clock_t seqClkStart, seqClkFinish, parClkStart, parClkFinish;

    double* A, *EigVectors, *Vec, *EigValues;
    double* d_EigVectors, * d_EigValues;
    int N, i, j, ij, iterations;

    // Parameters
    N = 128;
    iterations = 200;
    // Allocate memory for data
    A = (double*)malloc(sizeof(double) * N * N);
    Vec = (double*)malloc(N * N * sizeof(double));
    EigValues = (double*)malloc(N * sizeof(double));
    EigVectors = (double*)malloc(N * N * sizeof(double));
    d_EigValues = (double*)malloc(N * sizeof(double));
    d_EigVectors = (double*)malloc(N * N * sizeof(double));

    // Initialize A
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            ij = i * N + j;
            if (i == j) {
                A[ij] = i;
            }
            else {
                A[ij] = -0.25;
            }
        }
    }

    // CUDA eigen decomposition and timing
    parClkStart = clock();
    CUDAEigenDecomp(A, N, d_EigVectors, d_EigValues, iterations);
    parClkFinish = clock();

    // Sequential eigen decomposition and timing
    seqClkStart = clock();
    EigenDecomposition(A, N, EigVectors, EigValues, iterations);
    seqClkFinish = clock();

    // Verify computation of eigen vectors
    MatMult(A, EigVectors, N, Vec);

    printf("Verifying computation\n");
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            ij = i * N + j;
            assert(abs(EigVectors[ij] - d_EigVectors[ij]) <= 0.0001);
        }
        assert(abs(EigValues[i] - d_EigValues[i]) <= 0.0001);
    }
    printf("Parallel results consistent with sequential results\n");
    
    // Print timing stuff
    printf("Finished computation in seq:%lfms, par:%lfms\n", double(seqClkFinish - seqClkStart) / CLOCKS_PER_SEC * 1000.0, double(parClkFinish - parClkStart) / CLOCKS_PER_SEC * 1000.0);
    

    // Free allocated data
    free(A);
    free(EigVectors);
    free(d_EigVectors);
    free(Vec);
    free(EigValues);
    free(d_EigValues);

    printf("\ndone!\n");
    return 0;
}