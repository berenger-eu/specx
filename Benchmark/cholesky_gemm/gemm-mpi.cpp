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

#include "Scheduler/SpMultiPrioScheduler.hpp"

//////////////////////////////////////////////////////////////////////////////

void gemm(const int NbLoops, double matrixC[], const double matrixA[], const double matrixB[], const int inMatrixDim){
    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::NORMAL,
                    inMatrixDim, inMatrixDim, inMatrixDim, 1.0, matrixA, inMatrixDim,
                    matrixB, inMatrixDim,
                    1.0, matrixC, inMatrixDim );
    }
}

struct Coord{
    int i, j;
};

auto gemm(const int NbLoops, SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
          const Coord& processIdxInGrid, const int processBlockDim,
          const Coord& processGridDim, const int inMatrixDim, const int inBlockDim,
          const int nbGpu, const bool useMultiPrioScheduler){
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

    std::unique_ptr<SpBlas::Block[]> buffersA(new SpBlas::Block[processBlockDim*processBlockDim]);
    std::unique_ptr<SpBlas::Block[]> buffersB(new SpBlas::Block[processBlockDim*processBlockDim]);

#ifdef SPECX_COMPILE_WITH_CUDA
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(SpUtils::DefaultNumThreads(), nbGpu), std::move(scheduler));
#elif defined(SPECX_COMPILE_WITH_HIP)
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(SpUtils::DefaultNumThreads(), nbGpu), std::move(scheduler));    
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif

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

    std::vector<double> minMaxAvg(3);
    minMaxAvg[0] = std::numeric_limits<double>::max();
    minMaxAvg[1] = std::numeric_limits<double>::min();
    minMaxAvg[2] = 0;

    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        tg.computeOn(ce);

        SpTimer timer;

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


        tg.waitAllTasks();
        timer.stop();

#ifdef SPECX_COMPILE_WITH_CUDA
        for(int i = 0 ; i < processBlockDim ; ++i){
            for(int j = 0 ; j < processBlockDim ; ++j){
                tg.syncDataOnCpu(blocksC[i*processBlockDim+j]);
            }
        }
#endif

        tg.waitAllTasks();

        minMaxAvg[0] = std::min(minMaxAvg[0], timer.getElapsed());
        minMaxAvg[1] = std::max(minMaxAvg[1], timer.getElapsed());
        minMaxAvg[2] += timer.getElapsed();
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    ce.execOnWorkers([&handles, offsetWorker](auto id, auto type){
        if(type == SpWorkerTypes::Type::CUDA_WORKER){
            auto& hdl = handles[id-offsetWorker];
            CUBLAS_ASSERT(cublasDestroy(hdl));
        }
     });
#endif

    ce.stopIfNotAlreadyStopped();

    minMaxAvg[2] /= NbLoops;
    return minMaxAvg;
}

int main(int argc, char** argv){
    CLsimple args("Gemm", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int NbLoops = 10;
    args.addParameter<int>({"lp" ,"nbloops"}, "NbLoops", NbLoops, NbLoops);

    int MinMatrixSize = 16;
    args.addParameter<int>({"minms"}, "Min MatrixSize", MinMatrixSize, MinMatrixSize);

    int MaxMatrixSize = 16;
    args.addParameter<int>({"maxms"}, "Max MatrixSize", MaxMatrixSize, MaxMatrixSize);

    int MinBlockSize = 4;
    args.addParameter<int>({"minbs"}, "Min BlockSize", MinBlockSize, MinBlockSize);

    int MaxBlockSize = 4;
    args.addParameter<int>({"maxbs"}, "Max BlockSize", MaxBlockSize, MaxBlockSize);

    std::string outputDir = "./";
    args.addParameter<std::string>({"od"}, "outputdir", outputDir, outputDir);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return -1;
    }

#ifdef SPECX_COMPILE_WITH_CUDA  
    SpCudaUtils::PrintInfo(); 
    const int nbGpus = SpCudaUtils::GetNbDevices();
#elif defined(SPECX_COMPILE_WITH_HIP)
    SpHipUtils::PrintInfo();
    const int nbGpus = SpHipUtils::GetNbDevices();
#else    
    const int nbGpus = 0;
#endif 

    std::vector<std::vector<double>> allDurations;

    SpMpiBackgroundWorker::GetWorker().init();
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

    for(bool useMultiprio: std::vector<bool>{true, false}){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                    const bool printValues = (MatrixSize <= 16);
                    const bool checkAccuracy = (MatrixSize <= 16);
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

                    const auto minMaxAvg = gemm(NbLoops, blocksC.get(), blocksA.get(), blocksB.get(), processIdxInGrid,
                        processBlockDim, processGridDim, MatrixSize, BlockSize, idxGpu, useMultiprio);
                    allDurations.push_back(minMaxAvg);
                    std::cout << "     - Duration = " << minMaxAvg[0] << " " << minMaxAvg[1] << " " << minMaxAvg[2] << std::endl;
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
                    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrixC.get(), blocksC.get(), MatrixSize, BlockSize,
                                                                            0, 1, Psize);
                    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;
                }
            }
        }
    }

    // Print out csv
    if(Prank == 0){
        std::ofstream file(outputDir + "/gemm-mpi.csv");
        if(!file.is_open()){
            std::cerr << "Cannot open file " << outputDir + "/gemm-mpi.csv" << std::endl;
            return 1;
        }

        file << "NbGpu,MatrixSize,BlockSize,Multiprio,MinDuration,MaxDuration,AvgDuration" << std::endl;
        int idxDuration = 0;
        for(bool useMultiprio: std::vector<bool>{true, false}){
            for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
                for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                    for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                        file << idxGpu << "," << MatrixSize << "," << BlockSize << "," 
                            << (useMultiprio?"TRUE":"FALSE") << ","
                            << allDurations[idxDuration][0] << "," 
                            << allDurations[idxDuration][1] << "," 
                            << allDurations[idxDuration][2] << std::endl;
                        idxDuration += 1;
                    }
                }
            }
        }
    }

    return 0;
}
