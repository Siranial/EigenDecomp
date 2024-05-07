/*
 * MatrixOps.hpp
 *
 *  Created on: 15 apr. 2016
 *      Author: dharrison
 */

#ifndef MATRIXOPS_H_
#define MATRIXOPS_H_

void MatTranspose(double*, int, int, double*);
void VectorNormalization(double*, int);
void MatMult(double*, double*, int, double*);
void EigenDecomposition(double*, int, double*, double*, int);

#endif /* MATRIXOPS_H_ */