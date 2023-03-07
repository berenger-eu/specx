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

void gemm(double matrixC[], const double matrixA[], const double matrixB[], const int inMatrixDim){
    SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::NORMAL,
                    inMatrixDim, inMatrixDim, inMatrixDim, 1.0, matrixA, inMatrixDim,
                    matrixB, inMatrixDim,
                    1.0, matrixC, inMatrixDim );
}

void gemm(SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
                           const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;


     SpRuntime<> runtime;

#ifdef SPECX_COMPILE_WITH_CUDA
     std::vector<cublasHandle_t> handles(runtime.getNbCudaWorkers());
     const int offsetWorker = runtime.getNbCpuWorkers() + 1;
     runtime.execOnWorkers([&handles, offsetWorker, &runtime](){
         if(SpUtils::GetThreadType() == SpWorkerTypes::Type::CUDA_WORKER){
             assert(offsetWorker <= SpUtils::GetThreadId() && SpUtils::GetThreadId() < offsetWorker + runtime.getNbCudaWorkers());
             std::cout << "Worker " << SpUtils::GetThreadId() << " will now initiate cublas..." << std::endl;
             auto& hdl = handles[SpUtils::GetThreadId()-offsetWorker];
            CUBLAS_ASSERT(cublasCreate(&hdl));
            CUBLAS_ASSERT(cublasSetStream(hdl, SpCudaUtils::GetCurrentStream()));
         }
     });
#endif

    // Compute the blocks
    for(int i = 0 ; i < nbBlocks ; ++i){
        for(int j = 0 ; j < nbBlocks ; ++j){
            for(int k = 0 ; k < nbBlocks ; ++k){
                runtime.task(SpPriority(1), SpWrite(blocksC[i*nbBlocks+j]),
                        SpRead(blocksA[k*nbBlocks+j]), SpRead(blocksB[i*nbBlocks+k]),
                    SpCpu([inBlockDim](SpBlas::Block& blockC, const SpBlas::Block& blockA, const SpBlas::Block& blockB){
                        SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::NORMAL,
                                    inBlockDim, inBlockDim, inBlockDim, 1.0, blockA.values.get(), inBlockDim,
                                    blockB.values.get(), inBlockDim,
                                    1.0, blockC.values.get(), inBlockDim );
                    })
            #ifdef SPECX_COMPILE_WITH_CUDA
                  , SpCuda([inBlockDim, offsetWorker, &handles](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                      const SpDeviceDataView<const SpBlas::Block> paramB) {
                        // paramA.getRawPtr(), paramA.getRawSize()
                        const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
                        assert(idxCudaWorker < int(handles.size()));
                        const double alphaBeta = 1.0;
                        CUBLAS_ASSERT( cublasDgemm( handles[idxCudaWorker], CUBLAS_OP_N, CUBLAS_OP_N,
                                inBlockDim, inBlockDim, inBlockDim, &alphaBeta, (const double*)paramA.getRawPtr(), inBlockDim,
                                (const double*)paramB.getRawPtr(), inBlockDim,
                                &alphaBeta, (double*)paramC.getRawPtr(), inBlockDim ) );
                    })
            #endif
                ).setTaskName(std::string("GEMM -- (")+std::to_string(i)+","+std::to_string(j)+")");
            }
        }
    }

    for(int i = 0 ; i < nbBlocks ; ++i){
        for(int j = 0 ; j < nbBlocks ; ++j){
            runtime.task(SpWrite(blocksC[i*nbBlocks+j]),
                SpCpu([](SpBlas::Block& blockC){
                    // Move back to cpu
                })
            );
        }
    }

    runtime.waitAllTasks();
    runtime.stopAllThreads();
    runtime.generateDot("/tmp/graph.dot");

#ifdef SPECX_COMPILE_WITH_CUDA
    runtime.execOnWorkers([&handles, offsetWorker](){
        if(SpUtils::GetThreadType() == SpWorkerTypes::Type::CUDA_WORKER){
            auto& hdl = handles[SpUtils::GetThreadId()-offsetWorker];
            CUBLAS_ASSERT(cublasDestroy(hdl));
        }
     });
#endif
}

int main(){
    const int MatrixSize = 16;
    const int BlockSize = 2;
    const bool printValues = (MatrixSize <= 16);
    /////////////////////////////////////////////////////////
    auto matrixA = SpBlas::generateAMatrix(MatrixSize);
    auto matrixB = SpBlas::generateAMatrix(MatrixSize);
    auto matrixC = SpBlas::generateAMatrix(MatrixSize, 0);
    if(printValues){
        std::cout << "Matrix A:\n";
        SpBlas::printMatrix(matrixA.get(), MatrixSize);
        std::cout << "Matrix B:\n";
        SpBlas::printMatrix(matrixB.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    auto blocksA = SpBlas::matrixToBlock(matrixA.get(), MatrixSize, BlockSize);
    auto blocksB = SpBlas::matrixToBlock(matrixB.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks A:\n";
        SpBlas::printBlocks(blocksA.get(), MatrixSize, BlockSize);
        std::cout << "Blocks B:\n";
        SpBlas::printBlocks(blocksB.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    auto blocksC = SpBlas::matrixToBlock(matrixC.get(), MatrixSize, BlockSize);
    gemm(blocksC.get(), blocksA.get(), blocksB.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks after gemm C:\n";
        SpBlas::printBlocks(blocksC.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    gemm(matrixC.get(), matrixA.get(), matrixB.get(), MatrixSize);
    if(printValues){
        std::cout << "Matrix after gemm C:\n";
        SpBlas::printMatrix(matrixC.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrixC.get(), blocksC.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;

    return 0;
}
