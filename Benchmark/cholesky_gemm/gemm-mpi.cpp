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

#ifdef SPECX_COMPILE_WITH_CUDA
thread_local cublasHandle_t handle;
#endif

#ifdef SPECX_COMPILE_WITH_HIP
thread_local hipblasHandle_t handle;
#endif

template <int MaxNbDevices, const bool FavorLocality>
auto gemm(const int NbLoops, SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
          const Coord& processIdxInGrid, const int processBlockDim,
          const Coord& processGridDim, const int inMatrixDim, const int inBlockDim,
          const int nbGpu, const bool useMultiPrioScheduler){
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

    std::unique_ptr<SpBlas::Block[]> buffersA(new SpBlas::Block[processBlockDim*processBlockDim]);
    std::unique_ptr<SpBlas::Block[]> buffersB(new SpBlas::Block[processBlockDim*processBlockDim]);

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler<MaxNbDevices,FavorLocality>());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuGpuWorkers(SpUtils::DefaultNumThreads(), nbGpu), std::move(scheduler));
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif

#ifdef SPECX_COMPILE_WITH_CUDA
     ce.execOnWorkers([&ce](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
            CUBLAS_ASSERT(cublasCreate(&handle));
            CUBLAS_ASSERT(cublasSetStream(handle, SpCudaUtils::GetCurrentStream()));
         }
     });
#elif defined(SPECX_COMPILE_WITH_HIP)
     ce.execOnWorkers([&ce](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::HIP_WORKER){
            HIPBLAS_ASSERT(hipblasCreate(&handle));
            HIPBLAS_ASSERT(hipblasSetStream(handle, SpHipUtils::GetCurrentStream()));
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
                      , SpCuda([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                          const SpDeviceDataView<const SpBlas::Block> paramB) {
                            // paramA.getRawPtr(), paramA.getRawSize()
                            const double alphaBeta = 1.0;
                            CUBLAS_ASSERT( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    inBlockDim, inBlockDim, inBlockDim, &alphaBeta, (const double*)paramA.getRawPtr(), inBlockDim,
                                    (const double*)paramB.getRawPtr(), inBlockDim,
                                    &alphaBeta, (double*)paramC.getRawPtr(), inBlockDim ) );
                        })
                #endif
                #ifdef SPECX_COMPILE_WITH_HIP
                      , SpHip([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                          const SpDeviceDataView<const SpBlas::Block> paramB) {
                            // paramA.getRawPtr(), paramA.getRawSize()
                            const double alphaBeta = 1.0;
                            HIPBLAS_ASSERT(hipblasDgemm( handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
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
                              , SpCuda([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                                  const SpDeviceDataView<const SpBlas::Block> paramB) {
                                    // paramA.getRawPtr(), paramA.getRawSize()
                                    const double alphaBeta = 1.0;
                                    CUBLAS_ASSERT( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            inBlockDim, inBlockDim, inBlockDim, &alphaBeta, (const double*)paramA.getRawPtr(), inBlockDim,
                                            (const double*)paramB.getRawPtr(), inBlockDim,
                                            &alphaBeta, (double*)paramC.getRawPtr(), inBlockDim ) );
                                })
                        #endif
                        #ifdef SPECX_COMPILE_WITH_HIP
                              , SpHip([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                                  const SpDeviceDataView<const SpBlas::Block> paramB) {
                                    // paramA.getRawPtr(), paramA.getRawSize()
                                    const double alphaBeta = 1.0;
                                    HIPBLAS_ASSERT(hipblasDgemm( handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
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
                              , SpCuda([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                                  const SpDeviceDataView<const SpBlas::Block> paramB) {
                                    // paramA.getRawPtr(), paramA.getRawSize()
                                    const double alphaBeta = 1.0;
                                    CUBLAS_ASSERT( cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            inBlockDim, inBlockDim, inBlockDim, &alphaBeta, (const double*)paramA.getRawPtr(), inBlockDim,
                                            (const double*)paramB.getRawPtr(), inBlockDim,
                                            &alphaBeta, (double*)paramC.getRawPtr(), inBlockDim ) );
                                })
                        #endif
                        #ifdef SPECX_COMPILE_WITH_HIP
                              , SpHip([inBlockDim](SpDeviceDataView<SpBlas::Block> paramC, const SpDeviceDataView<const SpBlas::Block> paramA,
                                                  const SpDeviceDataView<const SpBlas::Block> paramB) {
                                    // paramA.getRawPtr(), paramA.getRawSize()
                                    const double alphaBeta = 1.0;
                                    HIPBLAS_ASSERT(hipblasDgemm( handle, HIPBLAS_OP_N, HIPBLAS_OP_N,
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

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
        for(int i = 0 ; i < processBlockDim ; ++i){
            for(int j = 0 ; j < processBlockDim ; ++j){
                tg.task(SpRead(blocksC[i*processBlockDim+j]),
                        [](const SpBlas::Block&){
                        });
            }
        }
        tg.waitAllTasks();
#endif

        minMaxAvg[0] = std::min(minMaxAvg[0], timer.getElapsed());
        minMaxAvg[1] = std::max(minMaxAvg[1], timer.getElapsed());
        minMaxAvg[2] += timer.getElapsed();
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    ce.execOnWorkers([](auto id, auto type){
        if(type == SpWorkerTypes::Type::CUDA_WORKER){
            CUBLAS_ASSERT(cublasDestroy(handle));
        }
     });
#elif SPECX_COMPILE_WITH_HIP
    ce.execOnWorkers([](auto id, auto type){
        if(type == SpWorkerTypes::Type::HIP_WORKER){
            HIPBLAS_ASSERT(hipDestroy(handle));
        }
     });
#endif

    ce.stopIfNotAlreadyStopped();

    minMaxAvg[2] /= NbLoops;

    // Add memory transfers info
    double totalAllocatedMemory = 0;
    double maxAllocatedMemory = 0;
    double deviceToHostTransfers = 0;
    double hostToDeviceTransfers = 0;
    double deviceToDeviceTransfers = 0;
#ifdef SPECX_COMPILE_WITH_CUDA
    for(int idxGpu = 0 ; idxGpu < nbGpu ; ++idxGpu){
        auto memInfo = SpCudaManager::Managers[idxGpu].getCounters();
        totalAllocatedMemory += memInfo[0].second/1e9;
        maxAllocatedMemory = std::max(maxAllocatedMemory, memInfo[1].second/1e9);
        deviceToHostTransfers += memInfo[2].second/1e9;
        hostToDeviceTransfers += memInfo[3].second/1e9;
        deviceToDeviceTransfers += memInfo[4].second/1e9;
        SpCudaManager::Managers[idxGpu].resetCounters();
    }
#elif defined(SPECX_COMPILE_WITH_HIP)
    for(int idxGpu = 0 ; idxGpu < nbGpu ; ++idxGpu){
        auto memInfo = SpHipManager::Managers[idxGpu].getCounters();
        totalAllocatedMemory += memInfo[0].second/1e9;
        maxAllocatedMemory = std::max(maxAllocatedMemory, memInfo[1].second/1e9);
        deviceToHostTransfers += memInfo[2].second/1e9;
        hostToDeviceTransfers += memInfo[3].second/1e9;
        deviceToDeviceTransfers += memInfo[4].second/1e9;
        SpHipManager::Managers[idxGpu].resetCounters();
    }
#endif
    minMaxAvg.push_back(totalAllocatedMemory/NbLoops);
    minMaxAvg.push_back(maxAllocatedMemory);
    minMaxAvg.push_back(deviceToHostTransfers/NbLoops);
    minMaxAvg.push_back(hostToDeviceTransfers/NbLoops);
    minMaxAvg.push_back(deviceToDeviceTransfers/NbLoops);

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

    assert(MinMatrixSize <= MaxMatrixSize);
    assert(MinBlockSize <= MaxBlockSize);

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
    const auto schedPairConf = std::vector<std::tuple<bool,bool>>{std::make_tuple(false, false),
                                                                 std::make_tuple(true, false),
                                                                 std::make_tuple(true, true)};

    SpMpiBackgroundWorker::GetWorker().init();
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();

    for(auto useMultiprioAndPairs: schedPairConf){
        for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
            for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
                    const bool useMultiprio = std::get<0>(useMultiprioAndPairs);
                    const bool useLocality = std::get<1>(useMultiprioAndPairs);

                    if(Prank == 0){
                        std::cout << "NbGpu = " << idxGpu << " MatrixSize = " << MatrixSize << 
                            " BlockSize = " << BlockSize << " Multiprio = " << (useMultiprio?"TRUE":"FALSE") << std::endl;
                    }
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

                    const auto minMaxAvg = (useLocality ? 
                                            gemm<8,true>(NbLoops, blocksC.get(), blocksA.get(), blocksB.get(), 
                                                MatrixSize, BlockSize, idxGpu, useMultiprio):
                                            gemm<8,false>(NbLoops, blocksC.get(), blocksA.get(), blocksB.get(), 
                                                MatrixSize, BlockSize, idxGpu, useMultiprio));
                    allDurations.push_back(minMaxAvg);
                    std::cout << Prank << "]     - Duration = " << minMaxAvg[0] << " " << minMaxAvg[1] << " " << minMaxAvg[2] << std::endl;
                    std::cout << Prank << "]     - Transfers = " << minMaxAvg[3] << " " << minMaxAvg[4] << " " << minMaxAvg[5] << " " << minMaxAvg[6] << " " << minMaxAvg[7] << std::endl;
                    if(printValues){
                        std::cout << "Blocks after gemm C:\n";
                        SpBlas::printBlocks(blocksC.get(), MatrixSize, BlockSize);
                    }
                    /////////////////////////////////////////////////////////
                    if(checkAccuracy){
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
    }

    // Print out csv
    if(Prank == 0){
        std::ofstream file(outputDir + "/gemm-mpi.csv");
        if(!file.is_open()){
            std::cerr << "Cannot open file " << outputDir + "/gemm-mpi.csv" << std::endl;
            return 1;
        }

        file << "NbGpu,MatrixSize,BlockSize,Multiprio,PrioPair,FavorLocality,MinDuration,MaxDuration,AvgDuration,TotalTransfer,MaxTransfer,DeviceToHostTransfer,HostToDeviceTransfer,DeviceToDeviceTransfer" << std::endl;
        int idxDuration = 0;
        for(auto useMultiprioAndPairs: schedPairConf){
            for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                    for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
                        const bool useMultiprio = std::get<0>(useMultiprioAndPairs);
                        const bool useLocality = std::get<1>(useMultiprioAndPairs);
                        file << idxGpu << "," << MatrixSize << "," << BlockSize << "," 
                            << (useMultiprio?"TRUE":"FALSE") << ","
                        << "FALSE" << ","
                        << (useLocality?"TRUE":"FALSE") << ","
                        << allDurations[idxDuration][0] << "," 
                        << allDurations[idxDuration][1] << "," 
                        << allDurations[idxDuration][2] << ","
                        << allDurations[idxDuration][3] << ","
                        << allDurations[idxDuration][4] << ","
                        << allDurations[idxDuration][5] << ","
                        << allDurations[idxDuration][6] << ","
                        << allDurations[idxDuration][7] << std::endl;
                        idxDuration += 1;
                    }
                }
            }
        }
    }

    return 0;
}
