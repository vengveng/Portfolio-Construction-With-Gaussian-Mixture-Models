#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>


// Compile with: gcc -O3 -ffast-math -o lib/portfolio_lib.so core/portfolio_lib.c -shared -fPIC -lm -lopenblas

__attribute__((visibility("default"), hot, flatten))
double max_sharpe_objective_and_jacobian(const double* __restrict w,
                                         const double* __restrict sigma,
                                         const double* __restrict mu,
                                         double* grad_out,
                                         size_t dim) {
    double num = cblas_ddot((int)dim, w, 1, mu, 1);
    double* y = (double*)malloc(dim * sizeof(double));
    if (!y) {
        return 1e6;
    }
    
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (int)dim, (int)dim,
                1.0, sigma, (int)dim, w, 1, 0.0, y, 1);

    double denom = cblas_ddot((int)dim, w, 1, y, 1);

    if (denom <= 1e-12) {
        free(y);
        return 1e6;
    }

    double sqrt_denom = sqrt(denom);
    double denom_3_2 = denom * sqrt_denom;

    for (size_t i = 0; i < dim; ++i) {
        grad_out[i] = -(mu[i] / sqrt_denom - num * y[i] / denom_3_2);
    }
    
    free(y);
    return -num / sqrt_denom;
}

// If you don't have BLAS linked, compile the code below instead. Comment out the cblas import.
// Compile with: gcc -O3 -ffast-math -o lib/portfolio_lib.so core/portfolio_lib.c -shared -fPIC -lm

// __attribute__((visibility("default"), hot, flatten))
// double max_sharpe_objective_and_jacobian(
//                                         const double* __restrict w,
//                                         const double* __restrict sigma,
//                                         const double* __restrict mu,
//                                         double* grad_out,
//                                         size_t dim) {

//     double num = 0.0;
//     double denom = 0.0;

//     for (size_t i = 0; i < dim; ++i) {
//         num += w[i] * mu[i];
//         for (size_t j = 0; j < dim; ++j) {
//             denom += w[i] * sigma[i * dim + j] * w[j];
//         }
//     }

//     if (denom <= 1e-12) return 1e6;

//     double sqrt_denom = sqrt(denom);
//     double denom_3_2 = denom * sqrt_denom;

//     for (size_t i = 0; i < dim; ++i) {
//         double dot_sigma = 0.0;
//         for (size_t j = 0; j < dim; ++j) {
//             dot_sigma += w[j] * sigma[i * dim + j];
//         }
//         grad_out[i] = -(mu[i] / sqrt_denom - num * dot_sigma / denom_3_2);
//     }

//     return -num / sqrt_denom;
// }