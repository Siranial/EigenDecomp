/*
 * Memory.c
 *
 *  Created on: 14 apr. 2016
 *      Author: dharrison
 */

#include <stdlib.h>


double* matrix1D(int np) {
    double* a;

    a = (double*)calloc(np, sizeof(double));

    return a;
}


double* matrix2D(int nm, int np) {
    double* m;

    m = (double*)calloc(nm * np, sizeof(double));

    return m;
}

void free2D(double* f, int nm) {
    delete[] f;
}