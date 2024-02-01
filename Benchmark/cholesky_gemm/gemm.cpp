#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <algorithm>
#include <memory>
#include <limits>

#include <clsimple.hpp>

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

void gemm(const int NbLoops, double matrixC[], const double matrixA[], const double matrixB[], const int inMatrixDim){
    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::NORMAL,
                    inMatrixDim, inMatrixDim, inMatrixDim, 1.0, matrixA, inMatrixDim,
                    matrixB, inMatrixDim,
                    1.0, matrixC, inMatrixDim );
    }
}

void gemm(const int NbLoops, SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
                           const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

     SpRuntime<> runtime;

#ifdef SPECX_COMPILE_WITH_CUDA
     std::vector<cublasHandle_t> handles(runtime.getNbCudaWorkers());
     const int offsetWorker = runtime.getNbCpuWorkers() + 1;
     runtime.execOnWorkers([&handles, offsetWorker, &runtime](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
             assert(offsetWorker <= id && id < offsetWorker + runtime.getNbCudaWorkers());
             SpDebugPrint() << "Worker " << id << " will now initiate cublas...";
             auto& hdl = handles[id-offsetWorker];
            CUBLAS_ASSERT(cublasCreate(&hdl));
            CUBLAS_ASSERT(cublasSetStream(hdl, SpCudaUtils::GetCurrentStream()));
         }
     });
#endif
    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        // Compute the blocks
        for(int i = 0 ; i < nbBlocks ; ++i){
            for(int j = 0 ; j < nbBlocks ; ++j){
                for(int k = 0 ; k < nbBlocks ; ++k){
                    runtime.task(SpPriority(1), SpCommutativeWrite(blocksC[i*nbBlocks+j]),
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
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    for(int i = 0 ; i < nbBlocks ; ++i){
        for(int j = 0 ; j < nbBlocks ; ++j){
            runtime.syncDataOnCpu(blocksC[i*nbBlocks+j]);
        }
    }
#endif

    runtime.waitAllTasks();

#ifdef SPECX_COMPILE_WITH_CUDA
    runtime.execOnWorkers([&handles, offsetWorker](auto id, auto type){
        if(type == SpWorkerTypes::Type::CUDA_WORKER){
            auto& hdl = handles[id-offsetWorker];
            CUBLAS_ASSERT(cublasDestroy(hdl));
        }
     });
#endif

    runtime.stopAllThreads();
    runtime.generateDot("/tmp/graph.dot");
    runtime.generateTrace("/tmp/gemm-simu.svg", false);
}

int main(int argc, char** argv){
    CLsimple args("Gemm", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int NbLoops;
    args.addParameter<int>({"lp" ,"nbloops"}, "NbLoops", NbLoops, 1);

    int MatrixSize;
    args.addParameter<int>({"ms"}, "MatrixSize", MatrixSize, 16);

    int BlockSize;
    args.addParameter<int>({"bs"}, "BlockSize", BlockSize, 2);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return -1;
    }

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
    gemm(NbLoops, blocksC.get(), blocksA.get(), blocksB.get(), MatrixSize, BlockSize);
    if(printValues){
        std::cout << "Blocks after gemm C:\n";
        SpBlas::printBlocks(blocksC.get(), MatrixSize, BlockSize);
    }
    /////////////////////////////////////////////////////////
    gemm(NbLoops, matrixC.get(), matrixA.get(), matrixB.get(), MatrixSize);
    if(printValues){
        std::cout << "Matrix after gemm C:\n";
        SpBlas::printMatrix(matrixC.get(), MatrixSize);
    }
    /////////////////////////////////////////////////////////
    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrixC.get(), blocksC.get(), MatrixSize, BlockSize);
    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;

    return 0;
}
