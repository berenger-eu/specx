#ifndef CHOLESKYFUNCTIONSWRAPPER_HPP
#define CHOLESKYFUNCTIONSWRAPPER_HPP

extern "C" {
void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
double dlange_(char *norm, int *m, int *n, const double *a, int *lda);
void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n, const double *alpha, const double *a, int *lda, double *b, int *ldb);
void dgemm_(char* transa, char* transb, int* m, int* n, int* k, double* alpha, const double* a, int* lda, const double* b, int* ldb, double *beta, double* c, int* ldc);
void dsyrk_(char *uplo, char *trans, int *n, int *k, double *alpha, const double *a, int *lda, double *beta, double *c, int *ldc);
void dgetrf_(int *m, int *n, double* a, int *lda, int *ipiv, int *info);
void dlaswp_(int *n, double* a, int *lda, int *k1, int *k2, int *ipiv, const int *incx);
}

namespace Cholesky
{

enum class Norm {
    M,
    // One norm
    ONE,
    // Infinity norm
    INF,
    // Frobenius norm
    FRO
};

inline char NormToChar(Norm selectedNorm){
    switch(selectedNorm) {
    case Norm::M:
        return 'M';
    case Norm::ONE:
        return '1';
    case Norm::INF:
        return 'I';
    case Norm::FRO:
        return 'F';
    }
    return '?';
}

enum class FillMode{
    LWPR,
    UPR
};

inline char FillModeToChar(FillMode mode){
    switch(mode) {
    case FillMode::LWPR:
        return 'L';
    case FillMode::UPR:
        return 'U';
    }
    return '?';
}

enum class Side{
    LEFT,
    RIGHT,
};

inline char SideToChar(Side side){
    switch(side) {
    case Side::LEFT:
        return 'L';
    case Side::RIGHT:
        return 'R';
    }
    return '?';
}

enum class DiagUnit{
    TRIANGULAR,
    NON_TRIANGULAR
};

inline char DiagUnitToChar(DiagUnit diag){
    switch(diag) {
    case DiagUnit::TRIANGULAR:
        return 'U';
    case DiagUnit::NON_TRIANGULAR:
        return 'N';
    }
    return '?';
}

enum class Transa{
    NORMAL,
    TRANSPOSE,
    CONJG
};

inline char TransaToChar(Transa transa){
    switch(transa) {
    case Transa::NORMAL:
        return 'N';
    case Transa::TRANSPOSE:
        return 'T';
    case Transa::CONJG:
        return 'C';
    }
    return '?';
}

inline void laswp(int n, double *a, int lda, int k1, int k2, int *perm, int incx) {
    dlaswp_(&n, a, &lda, &k1, &k2, perm, &incx);
}

inline void axpy(int n, double a, const double *x, int incx, double *y, int incy) {
    daxpy_(&n, &a, x, &incx, y, &incy);
}

inline double host_lange(Norm selectedNorm, int m, int n, const double *a, int lda){
    char fnorm = NormToChar(selectedNorm);
    return dlange_(&fnorm, &m, &n, a, &lda);
}

inline void potrf(FillMode uplo, int n, double* a, int lda) {
    char fuplo = FillModeToChar(uplo);
    int info;
    dpotrf_(&fuplo, &n, a, &lda, &info);

    if(info < 0){
        std::cout << "Error in potrf, argument " << (-info) << " is invalid." << std::endl;
    }
    else if(info > 0){
        std::cout << "Error in potrf, the leading minor of order " << info << " is not "
                     "positive definite, and the factorization could not be "
                     "completed." << std::endl;
    }
}

inline void trsm(
        Side side, FillMode uplo,
        Transa transa, DiagUnit diag,
        int m, int n,
        double alpha, const double* a, int lda,
        double* b, int ldb) {
    char fside = SideToChar(side);
    char fuplo = FillModeToChar(uplo);
    char ftransa = TransaToChar(transa);
    char fdiag = DiagUnitToChar(diag);
    dtrsm_(&fside, &fuplo, &ftransa, &fdiag, &m, &n, &alpha, a, &lda, b, &ldb);
}

inline void gemm(
        Transa transa, Transa transb,
        int m, int n, int k, double alpha, const double* a, int lda,
        const double* b, int ldb, double beta, double* c, int ldc) {
    char ftransa = TransaToChar(transa);
    char ftransb = TransaToChar(transb);
    dgemm_(&ftransa, &ftransb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}

inline void syrk(
        FillMode uplo, Transa trans,
        int n, int k, double alpha, const double* a, int lda,
        double beta, double* c, int ldc) {
    char fuplo = FillModeToChar(uplo);
    char ftrans = TransaToChar(trans);
    dsyrk_(&fuplo, &ftrans, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

inline int getrf(int m, int n, double* a, int lda, int *ipiv) {
    int info;
    dgetrf_(&m, &n, a, &lda, ipiv, &info);
    return info;
}

}

#endif
