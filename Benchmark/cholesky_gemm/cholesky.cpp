#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <algorithm>
#include <memory>
#include <limits>

#include "Utils/SpUtils.hpp"
#include "Legacy/SpRuntime.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"

#include "CholeskyFunctionsWrapper.hpp"


//////////////////////////////////////////////////////////////////////////////

void choleskyFactorizationMatrix(double matrix[], const int inMatrixDim){
    SpBlas::potrf( SpBlas::FillMode::LWPR, inMatrixDim, matrix, inMatrixDim );
}

void choleskyFactorization(SpBlas::Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    SpRuntime<> runtime;

#ifdef SPECX_COMPILE_WITH_CUDA
    struct CudaHandles{
        cublasHandle_t blasHandle;
        cusolverDnHandle_t solverHandle;
        size_t solverBuffSize;
        double* solverBuffer;
        cusolverDnParams_t params;
        int* cuinfo;
        double* val1;
        double* valminus1;
    };
     std::vector<CudaHandles> handles(runtime.getNbCudaWorkers());
     for(auto& hdl : handles){
        CUBLAS_ASSERT(cublasCreate(&hdl.blasHandle));
        CUBLAS_ASSERT(cublasSetStream(hdl.blasHandle, SpCudaUtils::GetCurrentStream()));
        CUSOLVER_ASSERT(cusolverDnCreate(&hdl.solverHandle));
        CUSOLVER_ASSERT(cusolverDnSetStream(hdl.solverHandle, SpCudaUtils::GetCurrentStream()));
        CUSOLVER_ASSERT(cusolverDnCreateParams(&hdl.params));
        CUSOLVER_ASSERT(cusolverDnPotrf_bufferSize(
                            hdl.solverHandle,
                            hdl.params,
                            CUBLAS_FILL_MODE_FULL,
                            inBlockDim,
                            CUDA_R_64F ,
                            nullptr,// TODO
                            inBlockDim,
                            CUDA_R_64F ,
                            &hdl.solverBuffSize));
        CUDA_ASSERT(cudaMalloc((void**)&hdl.solverBuffer , sizeof(double)*hdl.solverBuffSize));
        CUDA_ASSERT(cudaMalloc((void**)&hdl.cuinfo , sizeof(int)));

        CUDA_ASSERT(cudaMalloc((void**)&hdl.valminus1 , sizeof(double)));
        double minusone = -1;
        CUDA_ASSERT(cudaMemcpy(hdl.valminus1, &minusone, sizeof(double), cudaMemcpyHostToDevice));

        CUDA_ASSERT(cudaMalloc((void**)&hdl.val1 , sizeof(double)));
        double one = 1;
        CUDA_ASSERT(cudaMemcpy(hdl.val1, &one, sizeof(double), cudaMemcpyHostToDevice));
     }
     const int offsetWorker = runtime.getNbCpuWorkers() + 1;
#endif

    // Compute the blocks
    for(int k = 0 ; k < nbBlocks ; ++k){
        // TODO put for syrk and gemm ? const double rbeta = (j==0) ? beta : 1.0;
        // POTRF( RW A(k,k) )
        runtime.task(SpPriority(0), SpWrite(blocks[k*nbBlocks+k]),
            SpCpu([inBlockDim](SpBlas::Block& block){
                SpBlas::potrf( SpBlas::FillMode::LWPR, inBlockDim, block.values.get(), inBlockDim );
            })
#ifdef SPECX_COMPILE_WITH_CUDA
         , SpCuda([inBlockDim, offsetWorker, &handles](SpDeviceDataView<SpBlas::Block> param) {
            const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
            assert(idxCudaWorker < int(handles.size()));
            CUSOLVER_ASSERT(cusolverDnPotrf(
                handles[idxCudaWorker].solverHandle,
                handles[idxCudaWorker].params,
                CUBLAS_FILL_MODE_UPPER,
                inBlockDim,
                CUDA_R_64F ,
                (double*)param.getRawPtr(),
                inBlockDim,
                CUDA_R_64F ,
                handles[idxCudaWorker].solverBuffer,
                handles[idxCudaWorker].solverBuffSize,
                handles[idxCudaWorker].cuinfo));
        })
#endif
                ).setTaskName(std::string("potrf -- (W-")+std::to_string(k)+","+std::to_string(k)+")");

        for(int m = k + 1 ; m < nbBlocks ; ++m){
            // TRSM( R A(k,k), RW A(m, k) )
            runtime.task(SpPriority(1), SpRead(blocks[k*nbBlocks+k]), SpWrite(blocks[k*nbBlocks+m]),
            SpCpu([inBlockDim](const SpBlas::Block& blockA, SpBlas::Block& blockB){
                SpBlas::trsm( SpBlas::Side::RIGHT, SpBlas::FillMode::LWPR,
                                SpBlas::Transa::TRANSPOSE, SpBlas::DiagUnit::NON_UNIT_TRIANGULAR,
                                inBlockDim, inBlockDim, 1.0, blockA.values.get(), inBlockDim,
                                blockB.values.get(), inBlockDim );
            })
    #ifdef SPECX_COMPILE_WITH_CUDA
          , SpCuda([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                              SpDeviceDataView<SpBlas::Block> paramB) {
                const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
                assert(idxCudaWorker < int(handles.size()));
                CUBLAS_ASSERT( cublasDtrsm( handles[idxCudaWorker].blasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                        inBlockDim, inBlockDim, handles[idxCudaWorker].val1, (const double*)paramA.getRawPtr(), inBlockDim,
                        (double*)paramB.getRawPtr(), inBlockDim));
            })
    #endif
                    ).setTaskName(std::string("TRSM -- (R-")+std::to_string(k)+","+std::to_string(k)+") (W-"+std::to_string(m)+","+std::to_string(k)+")");
        }

        for(int n = k+1 ; n < nbBlocks ; ++n){
            // SYRK( R A(n,k), RW A(n, n) )
            runtime.task(SpPriority(1), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[n*nbBlocks+n]),
            SpCpu([inBlockDim](const SpBlas::Block& blockA, SpBlas::Block& blockC){
                SpBlas::syrk( SpBlas::FillMode::LWPR,
                                SpBlas::Transa::NORMAL,
                                inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                1.0, blockC.values.get(), inBlockDim );
            })
    #ifdef SPECX_COMPILE_WITH_CUDA
          , SpCuda([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                              SpDeviceDataView<SpBlas::Block> paramC) {
                // paramA.getRawPtr(), paramA.getRawSize()
                const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
                assert(idxCudaWorker < int(handles.size()));
                CUBLAS_ASSERT( cublasDsyrk( handles[idxCudaWorker].blasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                        inBlockDim, inBlockDim, handles[idxCudaWorker].valminus1, (const double*)paramA.getRawPtr(), inBlockDim,
                        handles[idxCudaWorker].val1, (double*)paramC.getRawPtr(), inBlockDim ) );
            })
    #endif
                    ).setTaskName(std::string("SYRK -- (R-")+std::to_string(n)+","+std::to_string(k)+") (W-"+std::to_string(n)+","+std::to_string(n)+")");

            for(int m = k+1 ; m < nbBlocks ; ++m){
                // GEMM( R A(m, k), R A(n, k), RW A(m, n))
                runtime.task(SpPriority(3), SpRead(blocks[k*nbBlocks+m]), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[m*nbBlocks+n]),
                SpCpu([inBlockDim](const SpBlas::Block& blockA, const SpBlas::Block& blockB, SpBlas::Block& blockC){
                    SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::TRANSPOSE,
                                    inBlockDim, inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                    blockB.values.get(), inBlockDim,
                                    1.0, blockC.values.get(), inBlockDim );
                })
        #ifdef SPECX_COMPILE_WITH_CUDA
              , SpCuda([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                                  const SpDeviceDataView<const SpBlas::Block> paramB, SpDeviceDataView<SpBlas::Block> paramC) {
                    const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
                    assert(idxCudaWorker < int(handles.size()));
                    CUBLAS_ASSERT( cublasDgemm( handles[idxCudaWorker].blasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                            inBlockDim, inBlockDim, inBlockDim, handles[idxCudaWorker].val1, (const double*)paramA.getRawPtr(), inBlockDim,
                            (const double*)paramB.getRawPtr(), inBlockDim,
                            handles[idxCudaWorker].valminus1, (double*)paramC.getRawPtr(), inBlockDim ) );
                })
        #endif
                        ).setTaskName(std::string("GEMM -- (R-")+std::to_string(m)+","+std::to_string(k)+")(R-"+std::to_string(n)+","+std::to_string(k)+")(W-"+std::to_string(m)+","+std::to_string(n)+")");
            }
        }
    }

    for(int i = 0 ; i < nbBlocks ; ++i){
        for(int j = 0 ; j < nbBlocks ; ++j){
            runtime.task(SpWrite(blocks[i*nbBlocks+j]),
                SpCpu([](SpBlas::Block& block){
                    // Move back to cpu
                })
            );
        }
    }

    runtime.waitAllTasks();
    runtime.stopAllThreads();
    runtime.generateDot("/tmp/graph.dot");

#ifdef SPECX_COMPILE_WITH_CUDA
     for(auto& hdl : handles){
        CUBLAS_ASSERT(cublasDestroy(hdl.blasHandle));
        CUSOLVER_ASSERT(cusolverDnDestroy(hdl.solverHandle));
        CUDA_ASSERT(cudaFree(hdl.solverBuffer));
        CUDA_ASSERT(cudaFree(hdl.val1));
        CUDA_ASSERT(cudaFree(hdl.valminus1));
     }
#endif
}

int main(){
    const int MatrixSize = 16;
    const int BlockSize = 2;
    const bool printValues = (MatrixSize <= 16);
    /////////////////////////////////////////////////////////
    auto matrix = SpBlas::generateMatrixLikeStarpu(MatrixSize);// SpBlas::generatePositiveDefinitMatrix(MatrixSize);
    if(printValues){
        std::cout << "Matrix:\n";
        SpBlas::printMatrix(matrix.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    auto blocks = SpBlas::matrixToBlock(matrix.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks:\n";
        SpBlas::printBlocks(blocks.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterCopy = SpBlas::diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after copy : " << errorAfterCopy << std::endl;
    /////////////////////////////////////////////////////////
    choleskyFactorization(blocks.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks after facto:\n";
        SpBlas::printBlocks(blocks.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    choleskyFactorizationMatrix(matrix.get(), MatrixSize);
    if(printValues){
        std::cout << "Matrix after facto:\n";
        SpBlas::printMatrix(matrix.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;

    return 0;
}
