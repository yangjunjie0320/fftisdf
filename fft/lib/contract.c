#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <cblas.h> // Requires linking with a BLAS library (e.g., -lopenblas or MKL)
// #include <omp.h>   // Include OpenMP header

void contract(
    complex double* x_kpt_out, // Output: nk * m * l
    const complex double* f_kpt, // Input:  nk * m * n
    const complex double* g_kpt, // Input:  nk * l * n
    const complex double* phase, // Input:  ns * nk
    int nk, int m, int n, int l
) {
    // Constants for BLAS calls
    const int ns = nk;
    const complex double alph = 1.0 + 0.0 * I;
    const complex double beta = 0.0 + 0.0 * I;

    // --- Allocate Temporaries ---
    complex double* t_kpt = (complex double*) malloc(nk * m * l * sizeof(complex double));
    complex double* t_spc = (complex double*) malloc(ns * m * l * sizeof(complex double));
    complex double* x_spc = (complex double*) malloc(ns * m * l * sizeof(complex double));

    // --- Step 1: Compute t_kpt = f_kpt.conj() @ g_kpt.T ---
    for (int k = 0; k < nk; ++k) {
        const complex double* fk = f_kpt + k * m * n;
        const complex double* gk = g_kpt + k * l * n;
        complex double*       tk = t_kpt + k * m * l;
        cblas_zgemm(
            CblasRowMajor, CblasConjNoTrans, CblasTrans,
            m, l, n, &alph, fk, n, gk, n, &beta, tk, l
        );
    }

    // --- Step 2: Compute t_spc = real(phase @ t_kpt_flat) ---
    // This is a single large BLAS call. Parallelism is best handled inside MKL/BLAS.
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        ns, m * l, nk, 
        &alph, phase, nk, t_kpt, m * l,      
        &beta, t_spc, m * l       
    );

    // --- Step 3: Compute x_spc = t_spc * t_spc ---
    // Element-wise square
    #pragma omp parallel for // Parallelize this loop
    for (int i = 0; i < ns * m * l; ++i) {
        x_spc[i] = t_spc[i] * t_spc[i];
    }

    // --- Step 4: Compute x_kpt = phase.conj().T @ x_spc ---
    // Matrix multiplication: phase.conj().T (nk, ns) @ x_spc (ns, m*l) -> x_kpt_out (nk, m*l)
    cblas_zgemm(
        CblasRowMajor, CblasConjTrans, CblasNoTrans,
        nk, m * l, ns,
        &alph, phase, nk, x_spc, m * l,
        &beta, x_kpt_out, m * l
    );

    // --- Cleanup ---
    free(t_kpt);
    free(t_spc);
    free(x_spc);
}