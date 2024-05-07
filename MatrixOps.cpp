/*
 * MatrixOps.cpp
 *
 *  Created on: 15 apr. 2016
 *      Author: dharrison
 */

#include <math.h>
#include <stdlib.h>
#include "Memory.hpp"


void MatMult(double* A, double* B, int N, double* result) {
    /*
     * This function performs matrix multiplication of two matrices A and B
     */
    int i, j, k;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            result[i * N + j] = 0.0;
        }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++)
                result[i * N + j] += A[i * N + k] * B[k * N + j];
        }
}


void MatTranspose(double* A, int rows, int columns, double* result) {
    /*
     * This function computes the transpose of matrix A
     */
    int i, j;

    for (i = 0; i < columns; i++)
        for (j = 0; j < rows; j++)
            result[i * rows + j] = A[j * columns + i];
            //result[i][j] = A[j][i];

}


void VectorNormalization(double* wp, int sizeVec) {
    int i;
    double sqrtwpwp = 0.0;

    for (i = 0; i < sizeVec; i++)
        sqrtwpwp += wp[i] * wp[i];

    sqrtwpwp = sqrt(sqrtwpwp);

    for (i = 0; i < sizeVec; i++)
        wp[i] = wp[i] / sqrtwpwp;
}


void EigenDecomposition(double* A, int N, double* EigVectors, double* EigValues, int iterations) {
    /*
     * This function computes the eigenvalues and eigenvectors
     * of a real, symmetric, N x N matrix A, using the QR algorithm.
     */

    int i, j, k, p, it;
    double* EigVecs, * Q, * EigVals, * result, * wp, * dumsum;
    double* R, * QT;
    double f;

    EigVecs = matrix2D(N, N);
    result = matrix2D(N, N);
    Q = matrix2D(N, N);
    EigVals = matrix2D(N, N);
    wp = matrix1D(N);
    dumsum = matrix1D(N);
    R = matrix2D(N, N);
    QT = matrix2D(N, N);

    //Initializing Ait, matrix containing eigenvalues as the diagonal
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            EigVals[i * N + j] = A[i * N + j];
        }
    }

    //Initializing Q and E for computation of eigenvectors
    for (i = 0; i < N; i++) {
        Q[i * N + i] = 1.0;
        EigVecs[i * N + i] = 1.0;
    }

    //Eigen decomposition iterations
    for (it = 0; it < iterations; it++) {
        //Gram-Schmidt decorrelation
        for (p = 0; p < N; p++) {
            for (i = 0; i < N; i++)
                wp[i] = EigVals[i * N + p];

            VectorNormalization(wp, N);

            for (i = 0; i < N; i++)
                dumsum[i] = 0.0;

            for (i = 0; i < N; i++) {
                for (j = 0; j < p; j++) {
                    f = 0.0;
                    for (k = 0; k < N; k++)
                        f += wp[k] * Q[k * N + j];

                    dumsum[i] += f * Q[i * N + j];
                }
            }

            for (i = 0; i < N; i++)
                wp[i] -= dumsum[i];

            VectorNormalization(wp, N);

            //Storing estimated rows of the inverse of the mixing matrix as columns in W
            for (i = 0; i < N; i++)
                Q[i * N + p] = wp[i];
        }

        MatTranspose(Q, N, N, QT);

        MatMult(QT, EigVals, N, R);

        MatMult(R, Q, N, EigVals);

        MatMult(EigVecs, Q, N, result);

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                EigVecs[i * N + j] = result[i * N + j];
            }
        }
    }

    //Set results
    for (i = 0; i < N; i++)
        EigValues[i] = EigVals[i * N + i];

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            EigVectors[i * N + j] = EigVecs[i * N + j];

    free2D(Q, N);
    free2D(EigVals, N);
    free2D(R, N);
    free2D(QT, N);
    free2D(EigVecs, N);
    free2D(result, N);
    delete[] wp;
    delete[] dumsum;
}