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

struct Coord{
    int i, j;
};

void gemm(SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
          const Coord& processIdxInGrid, const int processBlockDim,
          const Coord& processGridDim, const int inMatrixDim, const int inBlockDim){
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

    std::unique_ptr<SpBlas::Block[]> buffersA(new SpBlas::Block[processBlockDim*processBlockDim]);
    std::unique_ptr<SpBlas::Block[]> buffersB(new SpBlas::Block[processBlockDim*processBlockDim]);

    SpComputeEngine ce(SpWorkerTeamBuilder::DefaultTeamOfWorkers());
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
    tg.computeOn(ce);

#ifdef SPECX_COMPILE_WITH_CUDA
     std::vector<cublasHandle_t> handles(ce.getNbCudaWorkers());
     const int offsetWorker = ce.getNbCpuWorkers() + 1;
     ce.execOnWorkers([&handles, offsetWorker, &ce](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
             assert(offsetWorker <= id && id < offsetWorker + ce.getNbCudaWorkers());
             SpDebugPrint() << "Worker " << id << " will now initiate cublas...";
             auto& hdl = handles[id-offsetWorker];
            CUBLAS_ASSERT(cublasCreate(&hdl));
            CUBLAS_ASSERT(cublasSetStream(hdl, SpCudaUtils::GetCurrentStream()));
         }
     });
#endif

    for(int iProc = 0 ; iProc < processGridDim.i ; ++iProc){
        if(iProc != processIdxInGrid.i){
            const int dest = iProc*processGridDim.j + processIdxInGrid.j;
            for(int i = 0 ; i < processBlockDim ; ++i){
                for(int j = 0 ; j < processBlockDim ; ++j){
                    const int tag = i*processBlockDim+j;
                    tg.mpiSend(blocksB[i*processBlockDim+j], dest, tag);
                }
            }
        }
        else{
            [[maybe_unused]] const int dest = iProc*processGridDim.j + processIdxInGrid.j;
            assert(dest == Prank);
        }
    }
    for(int jProc = 0 ; jProc < processGridDim.j ; ++jProc){
        if(jProc != processIdxInGrid.j){
            const int dest = processIdxInGrid.i*processGridDim.j + jProc;
            for(int i = 0 ; i < processBlockDim ; ++i){
                for(int j = 0 ; j < processBlockDim ; ++j){
                    const int tag = i*processBlockDim+j;
                    tg.mpiSend(blocksA[i*processBlockDim+j], dest, tag);
                }
            }
        }
        else{
            [[maybe_unused]] const int dest = processIdxInGrid.i*processGridDim.j + jProc;
            assert(dest == Prank);
        }
    }

    // Compute the blocks
    for(int k = 0 ; k < processBlockDim ; ++k){
        for(int i = 0 ; i < processBlockDim ; ++i){
            for(int j = 0 ; j < processBlockDim ; ++j){
                tg.task(SpPriority(1), SpCommutativeWrite(blocksC[i*processBlockDim+j]),
                        SpRead(blocksA[k*processBlockDim+j]), SpRead(blocksB[i*processBlockDim+k]),
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

    for(int iProc = 0 ; iProc < processGridDim.i ; ++iProc){
        if(iProc != processIdxInGrid.i){
            const int recv = iProc*processGridDim.j + processIdxInGrid.j;
            for(int i = 0 ; i < processBlockDim ; ++i){
                for(int j = 0 ; j < processBlockDim ; ++j){
                    const int tag = i*processBlockDim+j;
                    tg.mpiRecv(buffersB[i*processBlockDim+j], recv, tag);
                }
            }

            for(int k = 0 ; k < processBlockDim ; ++k){
                for(int i = 0 ; i < processBlockDim ; ++i){
                    for(int j = 0 ; j < processBlockDim ; ++j){
                        tg.task(SpPriority(1), SpCommutativeWrite(blocksC[i*processBlockDim+j]),
                                SpRead(blocksA[k*processBlockDim+j]), SpRead(buffersB[i*processBlockDim+k]),
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
    }

    for(int jProc = 0 ; jProc < processGridDim.j ; ++jProc){
        if(jProc != processIdxInGrid.j){
            const int recv = processIdxInGrid.i*processGridDim.j + jProc;
            for(int i = 0 ; i < processBlockDim ; ++i){
                for(int j = 0 ; j < processBlockDim ; ++j){
                    const int tag = i*processBlockDim+j;
                    tg.mpiRecv(buffersA[i*processBlockDim+j], recv, tag);
                }
            }

            for(int k = 0 ; k < processBlockDim ; ++k){
                for(int i = 0 ; i < processBlockDim ; ++i){
                    for(int j = 0 ; j < processBlockDim ; ++j){
                        tg.task(SpPriority(1), SpCommutativeWrite(blocksC[i*processBlockDim+j]),
                                SpRead(buffersA[k*processBlockDim+j]), SpRead(blocksB[i*processBlockDim+k]),
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
    }

    for(int i = 0 ; i < processBlockDim ; ++i){
        for(int j = 0 ; j < processBlockDim ; ++j){
            tg.task(SpWrite(blocksC[i*processBlockDim+j]),
                SpCpu([](SpBlas::Block& blockC){
                    // Move back to cpu
                })
            );
        }
    }

    tg.waitAllTasks();

#ifdef SPECX_COMPILE_WITH_CUDA
    ce.execOnWorkers([&handles, offsetWorker](auto id, auto type){
        if(type == SpWorkerTypes::Type::CUDA_WORKER){
            auto& hdl = handles[id-offsetWorker];
            CUBLAS_ASSERT(cublasDestroy(hdl));
        }
     });
#endif

    ce.stopIfNotAlreadyStopped();
    tg.generateDot("/tmp/graph.dot");
}

int main(){
    SpMpiBackgroundWorker::GetWorker().init();
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

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

    const int sqrtPsize = std::round(std::sqrt(Psize));
    assert(sqrtPsize*sqrtPsize == Psize);
    Coord processGridDim{sqrtPsize, sqrtPsize};
    Coord processIdxInGrid{Prank/sqrtPsize, Prank%sqrtPsize};
    int processBlockDim = MatrixSize/BlockSize;

    std::cout << Prank << "] sqrtPsize = " << sqrtPsize << std::endl;
    std::cout << Prank << "] processGridDim = " << processGridDim.i << " " << processGridDim.j << std::endl;
    std::cout << Prank << "] processIdxInGrid = " << processIdxInGrid.i << " " << processIdxInGrid.j << std::endl;
    std::cout << Prank << "] processBlockDim = " << processBlockDim << std::endl;

    gemm(blocksC.get(), blocksA.get(), blocksB.get(), processIdxInGrid,
         processBlockDim, processGridDim, MatrixSize, BlockSize);
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
    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrixC.get(), blocksC.get(), MatrixSize, BlockSize,
                                                            0, 1, Psize);
    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;

    return 0;
}
