/*
 * EigenDecomposition.cu
 *
 *  Created on: March 18, 2024
 *      Author: Jonathan Adams
 *
 *      Implementation of the QR algorithm in CUDA C++ for computing
 *      eigen vectors and eigen values of a given real symmetric matrix
 */

#include "EigenDecomp.cuh"


void CUDAEigenDecomp(double* A, int N, double* EigVectors, double* EigValues, int iterations) {
    /*
    * This function computes the eigenvalues and eigenvectors
    * of a real, symmetric, N x N matrix A, using the QR algorithm
    * in parallel on device using CUDA C++.
    * Inputs:
    *  A: Matrix to be decomposed
    *  N: Size of A's sides
    *  EigVectors: Destination address for eigen vectors
    *  EigValues: Destination address for eigen values
    *  iterations: iterations to use for computation
    */

    double* d_A, * d_EigVectors, * d_EigValues, * d_EigVals;
    const unsigned int matrixSize = sizeof(double) * N * N;
    const unsigned int vectorSize = sizeof(double) * N;

    int i, j, k, p, it;
    double* d_Q, * d_result, * d_wp;
    double* d_R, * d_QT;

    // Allocate memory and copy A onto device
    assert(cudaMalloc((void **) &d_A, matrixSize) == cudaSuccess);
    assert(cudaMemcpy(d_A, A, matrixSize, cudaMemcpyHostToDevice) == cudaSuccess);
    
    // Allocate memory to device EigVectors and Values
    assert(cudaMalloc((void **) &d_EigVectors, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void **) &d_EigVals, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void**) &d_EigValues, vectorSize) == cudaSuccess);


    // Allocate memory to intermediates
    assert(cudaMalloc((void**)&d_result, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void**)&d_Q, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void**)&d_R, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void**)&d_QT, matrixSize) == cudaSuccess);
    assert(cudaMalloc((void**)&d_wp, vectorSize) == cudaSuccess);

    // Calculate grid/threads size for matrix operations
    const dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);
    const dim3 grid(ceil(N / (float)BLOCK_DIM), ceil(N / (float)BLOCK_DIM), 1);

    //Initializing Ait, matrix containing eigenvalues as the diagonal
    assert(cudaMemcpy(d_EigVals, d_A, matrixSize, cudaMemcpyDeviceToDevice) == cudaSuccess);

    //Initializing Q and E for computation of eigenvectors
    CUDASetDiag<<<1,grid.x>>>(d_EigVectors, N, 1.0);
    CUDASetDiag<<<1,grid.x>>>(d_Q, N, 1.0);

    //Eigen decomposition iterations
    for (it = 0; it < iterations; it++) {
        //Gram-Schmidt decorrelation
        for (p = 0; p < N; p++) {
            CUDACopyEigVals<<<1,grid.x>>>(d_wp, d_EigVals, N, p, 0);
            CUDASerialNormalize<<<1,1>>>(d_wp, N);
            CUDAWPDecrement<<<1,N>>>(d_wp, d_Q, N, p);
            CUDASerialNormalize<<<1,1>>>(d_wp, N);

            //Storing estimated rows of the inverse of the mixing matrix as columns in W
            CUDACopyEigVals<<<1,grid.x>>>(d_Q, d_wp, N, p, 1);
        }

        CUDASqMatTranspose<<<grid,threads>>>(d_Q, N, d_QT);
        CUDAMatMult<<<grid,threads>>>(d_QT, d_EigVals, N, d_R);
        CUDAMatMult<<<grid,threads>>>(d_R, d_Q, N, d_EigVals);
        CUDAMatMult<<<grid,threads>>>(d_EigVectors, d_Q, N, d_result);

        assert(cudaMemcpy(d_EigVectors, d_result, matrixSize, cudaMemcpyDeviceToDevice) == cudaSuccess);
    }
    
    CUDASetEigVals<<<1,grid.x>>>(d_EigValues, N, d_EigVals);

    //Set results
    assert(cudaMemcpy(EigVectors, d_EigVectors, matrixSize, cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(EigValues, d_EigValues, vectorSize, cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(d_A); cudaFree(d_EigVectors); cudaFree(d_EigValues);
    cudaFree(d_result); cudaFree(d_Q); cudaFree(d_R);
    cudaFree(d_QT); cudaFree(d_wp); cudaFree(d_EigVals);

    return;
}

__global__ void CUDASerialNormalize(double* A, int N) {
    /*
    * Normalizes matrix A destructively
    * Inputs:
    *  A: Matrix to normalize
    *  N: length of A
    */
    int i;
    double sqrtSum = 0.0;

    for (i = 0; i < N; i++)
        sqrtSum += A[i] * A[i];

    sqrtSum = sqrt(sqrtSum);

    for (i = 0; i < N; i++)
        A[i] = A[i] / sqrtSum;
}

__global__ void CUDAMatMult(double* A, double* B, int N, double* result) {
    /*
    * Mutliplies two square matrices with tiled algorithm
    * Inputs:
    *  A: Left matrix
    *  B: Right matrix
    *  N: Size of A/B's side length
    *  result: Matrix to store results
    *  
    */
    __shared__ double ABlock[BLOCK_DIM][BLOCK_DIM];
    __shared__ double BBlock[BLOCK_DIM][BLOCK_DIM];

    // Row i of matrix A
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int dest = row * N + col;

    double temp = 0.0;

    // Iterate blocks across all segments of shared memory
    for (int i = 0; i < (N - 1) / BLOCK_DIM + 1; i++) {
        // Load data into shared memory
        if (row < N && i * BLOCK_DIM + threadIdx.x < N)
            ABlock[threadIdx.y][threadIdx.x] = A[row * N + i * BLOCK_DIM + threadIdx.x];
        else
            ABlock[threadIdx.y][threadIdx.x] = 0.0;

        if (i * BLOCK_DIM + threadIdx.y < N && col < N)
            BBlock[threadIdx.y][threadIdx.x] = B[(i * BLOCK_DIM + threadIdx.y) * N + col];
        else
            BBlock[threadIdx.y][threadIdx.x] = 0.0;

        // Synchronize before computation
        __syncthreads();

        // Accumulate one tile of result from tiles of A and B in shared mem
        for (int j = 0; j < BLOCK_DIM; j++)
            temp += ABlock[threadIdx.y][j] * BBlock[j][threadIdx.x]; //eliminates bank conflict

        // Synchronize before new shared load
        __syncthreads();
    }
    // Store accumulated value
    if (row < N && col < N)
        result[dest] = temp;
    return;
}

__global__ void CUDASqMatTranspose(double* A, int N, double* result) {
    /*
    * Transposes a square matrix
    * Inputs:
    *  A: source matrix to transpose
    *  N: length of one of A's sides
    *  result: matrix to store results
    */

    // Initialize shared memory to reduce data transfer
    __shared__ double block[BLOCK_DIM][BLOCK_DIM+1];

    // Calculate memory location and load to shared memory
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((xIndex < N) && (yIndex < N))
    {
        unsigned int index_in = yIndex * N + xIndex;
        block[threadIdx.y][threadIdx.x] = A[index_in];
    }

    // synchronise to ensure all writes to block[][] have completed
    __syncthreads();
    
    // Calculate destination memory location and write
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((xIndex < N) && (yIndex < N))
    {
        unsigned int index_out = yIndex * N + xIndex;
        result[index_out] = block[threadIdx.x][threadIdx.y];
    }

    return;
}

__global__ void CUDACopyEigVals(double* dest, double* source, int N, int p, int direction) {
    /*
    * Copy N values from eig matrix source to dest
    * Inputs:
    *  dest: destination array
    *  source: source array
    *  N: # of eigen values per column
    *  p: current working column
    *  direction: 0 if source is N by N, 1 if dest is N by N.
    *             other double* is N length and stores eigenvals
    */
    int startIndex = threadIdx.x * BLOCK_DIM;
    int endIndex = startIndex + BLOCK_DIM;
    if (endIndex > N) endIndex = N;

    if (direction == 0) {
        for (int i = startIndex; i < endIndex; i++)
            dest[i] = source[i * N + p];
    }
    else {
        for (int i = startIndex; i < endIndex; i++)
            dest[i * N + p] = source[i];
    }

    return;
}

__global__ void CUDAWPDecrement(double* wp, double* Q, int N, int p) {
    /*
    * Performs the WP phase of eigen decomp
    * Inputs:
    *  wp: the wp matrix
    *  Q: the Q matrix
    *  N: size of wp
    *  p: current eigval column phase
    */
    int i, j, k;

    double dumsum = 0.0;
    double f;

    for (j = 0; j < p; j++) {
        f = 0.0;
        for (k = 0; k < N; k++)
            f += wp[k] * Q[k * N + j];

        dumsum += f * Q[threadIdx.x * N + j];
    }

    wp[threadIdx.x] -= dumsum;

    return;
}

__global__ void CUDASetDiag(double* dest, int N, double value) {
    /*
    * Sets N diagonal entries of dest to value
    * Inputs:
    *  dest: the destination matrix
    *  N: size of one of dest's sides
    *  value: value to set diagonalss to
    */
    int startIndex = threadIdx.x * BLOCK_DIM;
    int endIndex = startIndex + BLOCK_DIM;
    if (endIndex > N) endIndex = N;

    for (int i = startIndex; i < endIndex; i++)
        dest[i * N + i] = value;

    return;
}

__global__ void CUDASetEigVals(double* dest, int N, double* source) {
    /*
    * Extracts N eigenvalues from source to dest
    * Inputs:
    *  dest: the destination matrix
    *  N: size of one of dest's sides
    *  source: matrix to copy eigvals from
    */
    int startIndex = threadIdx.x * BLOCK_DIM;
    int endIndex = startIndex + BLOCK_DIM;
    if (endIndex > N) endIndex = N;

    for (int i = startIndex; i < endIndex; i++)
        dest[i] = source[i * N + i];

    return;
}


//UNUSED FUNCTIONS

__host__ void CUDANormalize(double* A, int N, int numThreads) {
    /*
    * Destructively normalizes the elements of A
    * Inputs:
    *  A: The vector to be normalized (data not preserved)
    *  N: length of A
    *  numThreads: number of threads to use for vector ops
    */
    double* d_sqSum, * sqSum;
    double sqrtSum = 0.0;

    size_t sumSize = sizeof(double) * numThreads;

    sqSum = (double*)malloc(sumSize);
    assert(cudaMalloc(&d_sqSum, sumSize) == cudaSuccess);

    CUDAVecSqSum << <1, numThreads >> > (A, N, d_sqSum);

    cudaDeviceSynchronize();

    assert(cudaMemcpy(sqSum, d_sqSum, sumSize, cudaMemcpyDeviceToHost) == cudaSuccess);

    for (int i = 0; i < numThreads; i++) {
        sqrtSum += sqSum[i];
    }

    cudaFree(d_sqSum); free(sqSum);

    sqrtSum = sqrt(sqrtSum);

    CUDADivideByConstant << <1, numThreads >> > (A, sqrtSum, N);
}

__global__ void CUDAVecSqSum(double* vector, int N, double* sqSum) {
    /*
    * Returns the sum of vector components
    * Inputs:
    *  vector: the vector to be summed
    *  N: length of vector
    *  sqSum: destination address of square sum
    */
    const int startIndex = threadIdx.x * BLOCK_DIM;
    int endIndex = startIndex + BLOCK_DIM;

    if (endIndex > N) endIndex = N;

    for (int i = startIndex; i < endIndex; i++)
        sqSum[threadIdx.x] += vector[i] * vector[i];

    return;
}

__global__ void CUDADivideByConstant(double* wp, double divisor, int N) {
    /*
    * Divides all elements of wp by sqsum
    * Inputs:
    *  wp: the vector to be divided by
    *  divisor: the divisor
    *  N: length of wp
    */
    const int startIndex = threadIdx.x * BLOCK_DIM;
    const int endIndex = startIndex + BLOCK_DIM;

    for (int i = startIndex; i < endIndex; i++) {
        if (i < N) wp[i] = wp[i] / divisor;
    }

    return;
}