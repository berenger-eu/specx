#ifndef CHOLESKYFUNCTIONSWRAPPER_HPP
#define CHOLESKYFUNCTIONSWRAPPER_HPP

#ifdef SPECX_COMPILE_WITH_CUDA
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif
#include <cblas.h>

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

#include <iostream>
#include <cmath>
#include <unistd.h>

namespace SpBlas
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
    UNIT_TRIANGULAR,
    NON_UNIT_TRIANGULAR
};

inline char DiagUnitToChar(DiagUnit diag){
    switch(diag) {
    case DiagUnit::UNIT_TRIANGULAR:
        return 'U';
    case DiagUnit::NON_UNIT_TRIANGULAR:
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


struct Block{
    int rowOffset;
    int colOffset;

    int nbRows;
    int nbCols;

    std::unique_ptr<double[]> values;

    Block() = default;
    Block(const Block&) = default;
    Block(Block&&) = default;
    Block& operator=(const Block&) = default;
    Block& operator=(Block&&) = default;

    /////////////////////////////////////////////////////////////

    struct DataDescr {
        int rowOffset = 0;
        int colOffset = 0;

        int nbRows = 0;
        int nbCols = 0;

        explicit DataDescr(){}
        DataDescr(int inRowOffset, int inColOffset, int inNbRows, int inNbCols)
            : rowOffset(inRowOffset), colOffset(inColOffset), nbRows(inNbRows), nbCols(inNbCols) {}
    };

    using DataDescriptor = DataDescr;

    std::size_t memmovNeededSize() const{
        return sizeof(double)*nbRows*nbCols;
    }

    template <class DeviceMemmov>
    auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
        double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
        mover.copyHostToDevice(doubleDevicePtr, values.get(), nbRows*nbCols*sizeof(double));
        return DataDescr{rowOffset, colOffset, nbRows, nbCols};
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
        double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
        mover.copyDeviceToHost(values.get(), doubleDevicePtr,  nbRows*nbCols*sizeof(double));
    }

    /////////////////////////////////////////////////////////////
#ifdef SPECX_COMPILE_WITH_MPI
    Block(SpDeserializer &deserializer)
        : rowOffset(deserializer.restore<decltype(rowOffset)>("rowOffset")),
          colOffset(deserializer.restore<decltype(colOffset)>("colOffset")),
          nbRows(deserializer.restore<decltype(nbRows)>("nbRows")),
          nbCols(deserializer.restore<decltype(nbCols)>("nbCols")){

        double* ptr = nullptr;
        deserializer.restore(ptr, "values");
        values.reset(ptr);
    }

    void serialize(SpSerializer &serializer) const {
        serializer.append(rowOffset, "rowOffset");
        serializer.append(colOffset, "colOffset");
        serializer.append(nbRows, "nbRows");
        serializer.append(nbCols, "nbCols");
        serializer.append(values.get(), nbRows*nbCols, "values");
    }
#endif
};

//////////////////////////////////////////////////////////////////////////////

std::unique_ptr<double[]> generatePositiveDefinitMatrix(const int inMatrixDim){
    std::unique_ptr<double[]> matrix(new double[inMatrixDim*inMatrixDim]());

    srand48(0);

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            matrix[idxRow*inMatrixDim+idxCol] = drand48();
            matrix[idxCol*inMatrixDim+idxRow] = matrix[idxRow*inMatrixDim+idxCol];
        }
    }

    for(int idxDiag = 0 ; idxDiag < inMatrixDim ; ++idxDiag){
        matrix[idxDiag*inMatrixDim+idxDiag] += inMatrixDim;
    }

    return matrix;
}

std::unique_ptr<double[]> generateMatrixLikeStarpu(const int inMatrixDim){
    std::unique_ptr<double[]> matrix(new double[inMatrixDim*inMatrixDim]());

    srand48(0);

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            matrix[idxCol*inMatrixDim+idxRow] = (1.0/double(1+idxRow+idxCol)) + ((idxRow == idxCol)?double(inMatrixDim):0.0);
        }
    }

    return matrix;
}


std::unique_ptr<double[]> generateAMatrix(const int inMatrixDim, const double inValue = 1){
    std::unique_ptr<double[]> matrix(new double[inMatrixDim*inMatrixDim]());

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            matrix[idxRow*inMatrixDim+idxCol] = inValue;
        }
    }

    return matrix;
}

std::unique_ptr<Block[]> matrixToBlock(double matrix[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    std::unique_ptr<Block[]> blocks(new Block[nbBlocks*nbBlocks]);

    // Init the blocks
    for(int m = 0 ; m < nbBlocks ; ++m){
        for(int n = 0 ; n < nbBlocks ; ++n){
            blocks[m*nbBlocks+n].rowOffset = m*inBlockDim;
            blocks[m*nbBlocks+n].colOffset = n*inBlockDim;
            blocks[m*nbBlocks+n].nbRows = std::min(inMatrixDim - m*inBlockDim, inBlockDim);
            blocks[m*nbBlocks+n].nbCols = std::min(inMatrixDim - n*inBlockDim, inBlockDim);
            blocks[m*nbBlocks+n].values.reset(new double[blocks[m*nbBlocks+n].nbRows * blocks[m*nbBlocks+n].nbCols]());
        }
    }

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            const int colIdxInBlock = idxCol%inBlockDim;

            blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock] = matrix[idxRow*inMatrixDim+idxCol];
        }
    }

    return blocks;
}

double diffMatrixBlocks(double matrix[], Block blocks[], const int inMatrixDim, const int inBlockDim,
                        const int Prank = 0, const int Psize = 1, const double alpha = 1.0){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    double error = 0;

    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            if(blockColIdx % Psize == Prank){
                const int colIdxInBlock = idxCol%inBlockDim;

                const double blockValue = blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock];
                const double matrixValue = alpha*matrix[idxRow*inMatrixDim+idxCol];

                error = std::max(error, std::abs(blockValue-matrixValue)/(std::abs(matrixValue)+std::numeric_limits<double>::epsilon()));
            }
        }
    }

    return error;
}

void printMatrix(const double matrix[], const int inMatrixDim){
    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        std::cout << idxRow << "]\t";
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            std::cout << matrix[idxRow*inMatrixDim+idxCol] << "\t";
        }
        std::cout << "\n";
    }
}

void printBlocks(const Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;
    for(int idxRow = 0 ; idxRow < inMatrixDim ; ++idxRow){
        std::cout << idxRow << "]\t";
        const int blockRowIdx = idxRow/inBlockDim;
        const int rowIdxInBlock = idxRow%inBlockDim;
        for(int idxCol = 0 ; idxCol < inMatrixDim ; ++idxCol ){
            const int blockColIdx = idxCol/inBlockDim;
            const int colIdxInBlock = idxCol%inBlockDim;

            const double blockValue = blocks[blockRowIdx*nbBlocks+blockColIdx].values[rowIdxInBlock*blocks[blockRowIdx*nbBlocks+blockColIdx].nbRows+colIdxInBlock];
            std::cout << blockValue << "\t";
        }
        std::cout << "\n";
    }
}

#ifdef SPECX_COMPILE_WITH_CUDA

inline const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

inline const char* cusolverGetStatusString(cusolverStatus_t status)
{
    switch(status)
    {
    case CUSOLVER_STATUS_SUCCESS : return "CUSOLVER_STATUS_SUCCESS The operation completed successfully.";
    case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED The cuSolver library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSolver routine, or an error in the hardware setup. To correct: call cusolverCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.";
    case CUSOLVER_STATUS_ALLOC_FAILED : return "CUSOLVER_STATUS_ALLOC_FAILED Resource allocation failed inside the cuSolver library. This is usually caused by a cudaMalloc() failure. To correct: prior to the function call, deallocate previously allocated memory as much as possible.";
    case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE An unsupported value or parameter was passed to the function (a negative vector size, for example). To correct: ensure that all the parameters being passed have valid values.";
    case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision. To correct: compile and run the application on a device with compute capability 2.0 or above.";
    case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons. To correct: check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed.";
    case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR An internal cuSolver operation failed. This error is usually caused by a cudaMemcpyAsync() failure. To correct: check that the hardware, an appropriate version of the driver, and the cuSolver library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function. To correct: check that the fields in descrA were set correctly.";
    }
    return "unknown error";
}

#define CUBLAS_ASSERT(X)\
{\
    cublasStatus_t ___resCuda = (X);\
    if ( CUBLAS_STATUS_SUCCESS != ___resCuda ){\
    printf("Error: fails, %s (%s line %d)\n", cublasGetStatusString(___resCuda), __FILE__, __LINE__ );\
    exit(1);\
    }\
    }

#define CUSOLVER_ASSERT(X)\
{\
    cusolverStatus_t ___resCuda = (X);\
    if ( CUSOLVER_STATUS_SUCCESS != ___resCuda ){\
    printf("Error: fails, %s (%s line %d)\n", SpBlas::cusolverGetStatusString(___resCuda), __FILE__, __LINE__ );\
    exit(1);\
    }\
    }

#endif

}

#endif
