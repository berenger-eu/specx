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

void choleskyFactorizationMatrix(const int NbLoops, double matrix[], const int inMatrixDim){
    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpBlas::potrf( SpBlas::FillMode::LWPR, inMatrixDim, matrix, inMatrixDim );
    }
}
#ifdef SPECX_COMPILE_WITH_CUDA
    struct CudaHandles{
        cublasHandle_t blasHandle;
        cusolverDnHandle_t solverHandle;
        int solverBuffSize;
        double* solverBuffer;
        int* cuinfo;
    };

thread_local CudaHandles handle;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
    struct HipHandles{
        hipblasHandle_t blasHandle;
        hipsolverHandle_t solverHandle;
        int solverBuffSize;
        double* solverBuffer;
        int* cuinfo;
    };

thread_local HipHandles handle;
#endif

template <int MaxNbDevices, const bool FavorLocality>
auto choleskyFactorization(const int NbLoops, SpBlas::Block blocksInput[], const int inMatrixDim, const int inBlockDim,
                           const int nbGpu, const bool useMultiPrioScheduler, const bool usePrioPairs){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;
    const int Psize = SpMpiUtils::GetMpiSize();
    const int Prank = SpMpiUtils::GetMpiRank();

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
    ce.execOnWorkers([&ce, inBlockDim, &blocksInput](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
             SpDebugPrint() << "Worker " << id << " will now initiate cublas...";
             CUBLAS_ASSERT(cublasCreate(&handle.blasHandle));
             CUBLAS_ASSERT(cublasSetStream(handle.blasHandle, SpCudaUtils::GetCurrentStream()));
             CUSOLVER_ASSERT(cusolverDnCreate(&handle.solverHandle));
             CUSOLVER_ASSERT(cusolverDnSetStream(handle.solverHandle, SpCudaUtils::GetCurrentStream()));
             CUSOLVER_ASSERT(cusolverDnDpotrf_bufferSize(
                                 handle.solverHandle,
                                 CUBLAS_FILL_MODE_LOWER,
                                 inBlockDim,
                                 blocksInput[0].values.get(),
                                 inBlockDim,
                                 &handle.solverBuffSize));
             CUDA_ASSERT(cudaMalloc((void**)&handle.solverBuffer , sizeof(double)*handle.solverBuffSize));
             CUDA_ASSERT(cudaMalloc((void**)&handle.cuinfo , sizeof(int)));
         }
     });
#endif
#ifdef SPECX_COMPILE_WITH_HIP
        ce.execOnWorkers([&ce, inBlockDim, &blocksInput](auto id, auto type){
            assert(id == SpUtils::GetThreadId());
            assert(type == SpUtils::GetThreadType());
            if(type == SpWorkerTypes::Type::HIP_WORKER){
                SpDebugPrint() << "Worker " << id << " will now initiate hipblas...";
                HIPBLAS_ASSERT(hipblasCreate(&handle.blasHandle));
                HIPBLAS_ASSERT(hipblasSetStream(handle.blasHandle, SpHipUtils::GetCurrentStream()));
                HIPSOLVER_ASSERT(hipsolverCreate(&handle.solverHandle));
                HIPSOLVER_ASSERT(hipsolverSetStream(handle.solverHandle, SpHipUtils::GetCurrentStream()));
                HIPSOLVER_ASSERT(hipsolverDpotrf_bufferSize(
                                    handle.solverHandle,
                                    HIPBLAS_FILL_MODE_LOWER,
                                    inBlockDim,
                                    blocksInput[0].values.get(),
                                    inBlockDim,
                                    &handle.solverBuffSize));
                HIP_ASSERT(hipMalloc((void**)&handle.solverBuffer , sizeof(double)*handle.solverBuffSize));
                HIP_ASSERT(hipMalloc((void**)&handle.cuinfo , sizeof(int)));
            }
        });
#endif

    int potrfPriority = 2;
    int trsmPriority = 1;
    int syrkPriority = 1;
    int gemmPriority = 0;
    if(usePrioPairs){
        potrfPriority = SpMultiPrioScheduler<MaxNbDevices,FavorLocality>::GeneratePriorityWorkerPair(2, 0);
        trsmPriority = SpMultiPrioScheduler<MaxNbDevices,FavorLocality>::GeneratePriorityWorkerPair(1, 0);
        syrkPriority = SpMultiPrioScheduler<MaxNbDevices,FavorLocality>::GeneratePriorityWorkerPair(1, 0);
        gemmPriority = SpMultiPrioScheduler<MaxNbDevices,FavorLocality>::GeneratePriorityWorkerPair(0, 2);
    }

    std::vector<double> minMaxAvg(3);
    minMaxAvg[0] = std::numeric_limits<double>::max();
    minMaxAvg[1] = std::numeric_limits<double>::min();
    minMaxAvg[2] = 0;

     for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        tg.computeOn(ce);

        auto blocks = duplicateBlocks(nbBlocks*nbBlocks, blocksInput);

        SpTimer timer;

        // Compute the blocks
        for(int k = 0 ; k < nbBlocks ; ++k){
            // TODO put for syrk and gemm ? const double rbeta = (j==0) ? beta : 1.0;
            // POTRF( RW A(k,k) )
            if(k % Psize == Prank){
                tg.task(SpPriority(potrfPriority), SpWrite(blocks[k*nbBlocks+k]),
                    SpCpu([inBlockDim](SpBlas::Block& block){
                        SpBlas::potrf( SpBlas::FillMode::LWPR, inBlockDim, block.values.get(), inBlockDim );
                    })
    #ifdef SPECX_COMPILE_WITH_CUDA
                 , SpCuda([inBlockDim](SpDeviceDataView<SpBlas::Block> param) {
                    CUSOLVER_ASSERT(cusolverDnDpotrf(
                        handle.solverHandle,
                        CUBLAS_FILL_MODE_LOWER,
                        inBlockDim,
                        (double*)param.getRawPtr(),
                        inBlockDim,
                        handle.solverBuffer,
                        handle.solverBuffSize,
                        handle.cuinfo));
    #ifndef NDEBUG
                    int info;
                    CUDA_ASSERT(cudaMemcpy(&info, handle.cuinfo, sizeof(int), cudaMemcpyDeviceToHost));
                    assert(info >= 0);
    #endif
                })
    #endif
    #ifdef SPECX_COMPILE_WITH_HIP
                    , SpHip([inBlockDim](SpDeviceDataView<SpBlas::Block> param) {
                        HIPSOLVER_ASSERT(hipsolverDpotrf(
                            handle.solverHandle,
                            HIPBLAS_FILL_MODE_LOWER,
                            inBlockDim,
                            (double*)param.getRawPtr(),
                            inBlockDim,
                            handle.solverBuffer,
                            handle.solverBuffSize,
                            handle.cuinfo));
    #ifndef NDEBUG
                        int info;
                        HIP_ASSERT(hipMemcpy(&info, handle.cuinfo, sizeof(int), hipMemcpyDeviceToHost));
                        assert(info >= 0);
    #endif
                    })
    #endif
                    ).setTaskName(std::string("potrf -- (W-")+std::to_string(k)+","+std::to_string(k)+")");

                tg.mpiBroadcastSend(blocks[k*nbBlocks+k], Prank);
            }
            else{
                tg.mpiBroadcastRecv(blocks[k*nbBlocks+k], k % Psize);
            }

            for(int m = k + 1 ; m < nbBlocks ; ++m){
                // TRSM( R A(k,k), RW A(m, k) )
                if(m % Psize == Prank){
                    tg.task(SpPriority(trsmPriority), SpRead(blocks[k*nbBlocks+k]), SpWrite(blocks[k*nbBlocks+m]),
                    SpCpu([inBlockDim](const SpBlas::Block& blockA, SpBlas::Block& blockB){
                        SpBlas::trsm( SpBlas::Side::RIGHT, SpBlas::FillMode::LWPR,
                                        SpBlas::Transa::TRANSPOSE, SpBlas::DiagUnit::NON_UNIT_TRIANGULAR,
                                        inBlockDim, inBlockDim, 1.0, blockA.values.get(), inBlockDim,
                                        blockB.values.get(), inBlockDim );
                    })
        #ifdef SPECX_COMPILE_WITH_CUDA
                  , SpCuda([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                      SpDeviceDataView<SpBlas::Block> paramB) {
                        const double one = 1;
                        CUBLAS_ASSERT( cublasDtrsm( handle.blasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                                inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                (double*)paramB.getRawPtr(), inBlockDim));
                    })
        #endif
        #ifdef SPECX_COMPILE_WITH_HIP
                    , SpHip([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                        SpDeviceDataView<SpBlas::Block> paramB) {
                        const double one = 1;
                        HIPBLAS_ASSERT(hipblasDtrsm( handle.blasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER,
                                                    HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT,
                                inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                (double*)paramB.getRawPtr(), inBlockDim));
                    })
        #endif
                        ).setTaskName(std::string("TRSM -- (R-")+std::to_string(k)+","+std::to_string(k)+") (W-"+std::to_string(m)+","+std::to_string(k)+")");

                    tg.mpiBroadcastSend(blocks[k*nbBlocks+m], Prank);
                }
                else{
                    tg.mpiBroadcastRecv(blocks[k*nbBlocks+m], m % Psize);
                }
            }

            for(int n = k+1 ; n < nbBlocks ; ++n){
                // SYRK( R A(n,k), RW A(n, n) )
                if(n % Psize == Prank){
                    tg.task(SpPriority(syrkPriority), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[n*nbBlocks+n]),
                    SpCpu([inBlockDim](const SpBlas::Block& blockA, SpBlas::Block& blockC){
                        SpBlas::syrk( SpBlas::FillMode::LWPR,
                                        SpBlas::Transa::NORMAL,
                                        inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                        1.0, blockC.values.get(), inBlockDim );
                    })
        #ifdef SPECX_COMPILE_WITH_CUDA
                  , SpCuda([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                      SpDeviceDataView<SpBlas::Block> paramC) {
                        // paramA.getRawPtr(), paramA.getRawSize()
                        const double one = 1;
                        const double minusone = -1;
                        CUBLAS_ASSERT( cublasDsyrk( handle.blasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                inBlockDim, inBlockDim, &minusone, (const double*)paramA.getRawPtr(), inBlockDim,
                                &one, (double*)paramC.getRawPtr(), inBlockDim ) );
                    })
        #endif
        #ifdef SPECX_COMPILE_WITH_HIP
                    , SpHip([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                        SpDeviceDataView<SpBlas::Block> paramC) {
                        // paramA.getRawPtr(), paramA.getRawSize()
                        const double one = 1;
                        const double minusone = -1;
                        HIPBLAS_ASSERT(hipblasDsyrk( handle.blasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N,
                                inBlockDim, inBlockDim, &minusone, (const double*)paramA.getRawPtr(), inBlockDim,
                                &one, (double*)paramC.getRawPtr(), inBlockDim ) );
                    })
        #endif
                        ).setTaskName(std::string("SYRK -- (R-")+std::to_string(n)+","+std::to_string(k)+") (W-"+std::to_string(n)+","+std::to_string(n)+")");

                    tg.mpiBroadcastSend(blocks[n*nbBlocks+n], Prank);
                }
                else{
                    tg.mpiBroadcastRecv(blocks[n*nbBlocks+n], n % Psize);
                }

                for(int m = n+1 ; m < nbBlocks ; ++m){
                    // GEMM( R A(m, k), R A(n, k), RW A(m, n))
                    if(n % Psize == Prank){
                        tg.task(SpPriority(gemmPriority), SpRead(blocks[k*nbBlocks+m]), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[m*nbBlocks+n]),
                        SpCpu([inBlockDim](const SpBlas::Block& blockA, const SpBlas::Block& blockB, SpBlas::Block& blockC){
                            SpBlas::gemm( SpBlas::Transa::NORMAL, SpBlas::Transa::TRANSPOSE,
                                            inBlockDim, inBlockDim, inBlockDim, -1.0, blockA.values.get(), inBlockDim,
                                            blockB.values.get(), inBlockDim,
                                            1.0, blockC.values.get(), inBlockDim );
                        })
            #ifdef SPECX_COMPILE_WITH_CUDA
                      , SpCuda([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                          const SpDeviceDataView<const SpBlas::Block> paramB, SpDeviceDataView<SpBlas::Block> paramC) {
                            const double one = 1;
                            const double minusone = -1;
                            CUBLAS_ASSERT( cublasDgemm( handle.blasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                    inBlockDim, inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                    (const double*)paramB.getRawPtr(), inBlockDim,
                                    &minusone, (double*)paramC.getRawPtr(), inBlockDim ) );
                        })
            #endif
            #ifdef SPECX_COMPILE_WITH_HIP
                        , SpHip([inBlockDim](const SpDeviceDataView<const SpBlas::Block> paramA,
                                            const SpDeviceDataView<const SpBlas::Block> paramB, SpDeviceDataView<SpBlas::Block> paramC) {
                            const double one = 1;
                            const double minusone = -1;
                            HIPBLAS_ASSERT(hipblasDgemm( handle.blasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                                    inBlockDim, inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                    (const double*)paramB.getRawPtr(), inBlockDim,
                                    &minusone, (double*)paramC.getRawPtr(), inBlockDim ) );
                        })
            #endif
                            ).setTaskName(std::string("GEMM -- (R-")+std::to_string(m)+","+std::to_string(k)+")(R-"+std::to_string(n)+","+std::to_string(k)+")(W-"+std::to_string(m)+","+std::to_string(n)+")");

                        tg.mpiBroadcastSend(blocks[m*nbBlocks+n], Prank);
                    }
                    else{
                        tg.mpiBroadcastRecv(blocks[m*nbBlocks+n], n % Psize);
                    }
                }
            }
        }

        tg.waitAllTasks();
        timer.stop();

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
        for(int i = 0 ; i < nbBlocks ; ++i){
            for(int j = 0 ; j < nbBlocks ; ++j){
                if(j % Psize == Prank){
                    tg.task(SpRead(blocks[i*nbBlocks+j]),
                            [](const SpBlas::Block&){
                            });
                }
            }
        }
        tg.waitAllTasks();
#endif

        minMaxAvg[0] = std::min(minMaxAvg[0], timer.getElapsed());
        minMaxAvg[1] = std::max(minMaxAvg[1], timer.getElapsed());
        minMaxAvg[2] += timer.getElapsed();

        if(idxLoop == NbLoops-1){
            // Copy block back
            for(int idxBlock = 0 ; idxBlock < nbBlocks*nbBlocks ; ++idxBlock){
                std::copy(blocks[idxBlock].values.get(), 
                            blocks[idxBlock].values.get()+blocks[idxBlock].nbRows*blocks[idxBlock].nbCols, 
                            blocksInput[idxBlock].values.get());
            }
        }
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    ce.execOnWorkers([](auto id, auto type){
        if(type == SpWorkerTypes::Type::CUDA_WORKER){
            CUBLAS_ASSERT(cublasDestroy(handle.blasHandle));
            CUSOLVER_ASSERT(cusolverDnDestroy(handle.solverHandle));
            CUDA_ASSERT(cudaFree(handle.solverBuffer));
        }
     });
#elif defined(SPECX_COMPILE_WITH_HIP)
    ce.execOnWorkers([](auto id, auto type){
        if(type == SpWorkerTypes::Type::HIP_WORKER){
            HIPBLAS_ASSERT(hipblasDestroy(handle.blasHandle));
            HIPSOLVER_ASSERT(hipsolverDestroy(handle.solverHandle));
            HIP_ASSERT(hipFree(handle.solverBuffer));
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
        maxAllocatedMemory = std::max(maxAllocatedMemory, memInfo[2].second/1e9);
        deviceToHostTransfers += memInfo[3].second/1e9;
        hostToDeviceTransfers += memInfo[4].second/1e9;
        deviceToDeviceTransfers += memInfo[5].second/1e9;
        SpCudaManager::Managers[idxGpu].resetCounters();
    }
#elif defined(SPECX_COMPILE_WITH_HIP)
    for(int idxGpu = 0 ; idxGpu < nbGpu ; ++idxGpu){
        auto memInfo = SpHipManager::Managers[idxGpu].getCounters();
        totalAllocatedMemory += memInfo[0].second/1e9;
        maxAllocatedMemory = std::max(maxAllocatedMemory, memInfo[2].second/1e9);
        deviceToHostTransfers += memInfo[3].second/1e9;
        hostToDeviceTransfers += memInfo[4].second/1e9;
        deviceToDeviceTransfers += memInfo[5].second/1e9;
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
    CLsimple args("Cholesky", argc, argv);

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

    SpMpiBackgroundWorker::GetWorker().init();
    [[maybe_unused]] const int Psize = SpMpiUtils::GetMpiSize();
    [[maybe_unused]] const int Prank = SpMpiUtils::GetMpiRank();


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
    const auto schedPairConf = std::vector<std::tuple<bool,bool,bool>>{std::make_tuple(false, false,false),
                                                                 std::make_tuple(true, false, false),
                                                                 std::make_tuple(true, true, false),
                                                                 std::make_tuple(true, true, true),
                                                                 std::make_tuple(true, false, true)};

    for(auto useMultiprioAndPairs: schedPairConf){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                    const bool useMultiprio = std::get<0>(useMultiprioAndPairs);
                    const bool usePrioPairs = std::get<1>(useMultiprioAndPairs);
                    const bool useLocality = std::get<2>(useMultiprioAndPairs);

                    if(Prank == 0){
                        std::cout << "NbGpu = " << idxGpu << " MatrixSize = " << MatrixSize
                              << " BlockSize = " << BlockSize << " Multiprio = " << useMultiprio
                              << " Use pairs = " << usePrioPairs
                              << " Favor Loc = " << useLocality << std::endl;
                    }
                    
                    const bool printValues = (MatrixSize <= 16);
                    const bool checkValues = (MatrixSize <= 16);
                    /////////////////////////////////////////////////////////
                    auto matrix = SpBlas::generateMatrixLikeStarpu(MatrixSize);// SpBlas::generatePositiveDefinitMatrix(MatrixSize);
                    if(printValues && Prank == 0){
                        std::cout << "Matrix:\n";
                        SpBlas::printMatrix(matrix.get(), MatrixSize);
                    }
                    /////////////////////////////////////////////////////////
                    auto blocks = SpBlas::matrixToBlock(matrix.get(), MatrixSize, BlockSize);
                    if(printValues && Prank == 0){
                        std::cout << "Blocks:\n";
                        SpBlas::printBlocks(blocks.get(), MatrixSize, BlockSize);
                    }
                    /////////////////////////////////////////////////////////
                    if(checkValues){
                        const double errorAfterCopy = SpBlas::diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
                        std::cout << "Accuracy after copy : " << errorAfterCopy << std::endl;
                    }
                    /////////////////////////////////////////////////////////
                    const auto minMaxAvg = (useLocality ? 
                                                choleskyFactorization<8,true>(NbLoops, blocks.get(), MatrixSize, BlockSize, 
                                                                idxGpu, useMultiprio, usePrioPairs) :
                                                choleskyFactorization<8,false>(NbLoops, blocks.get(), MatrixSize, BlockSize, 
                                                                idxGpu, useMultiprio, usePrioPairs));
                    allDurations.push_back(minMaxAvg);
                    std::cout << Prank << "]    - Duration = " << minMaxAvg[0] << " " << minMaxAvg[1] << " " << minMaxAvg[2] << std::endl;
                    std::cout << Prank << "]    - Transfers = " << minMaxAvg[3] << " " << minMaxAvg[4] << " " << minMaxAvg[5] << " " << minMaxAvg[6] << " " << minMaxAvg[7] << std::endl;
                    if(printValues && Prank == 0){
                        std::cout << "Blocks after facto:\n";
                        SpBlas::printBlocks(blocks.get(), MatrixSize, BlockSize);
                    }
                    if(checkValues){
                        /////////////////////////////////////////////////////////
                        choleskyFactorizationMatrix(NbLoops, matrix.get(), MatrixSize);
                        if(printValues && Prank == 0){
                            std::cout << "Matrix after facto:\n";
                            SpBlas::printMatrix(matrix.get(), MatrixSize);
                        }
                        /////////////////////////////////////////////////////////
                        const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize,
                                                                                Prank, Psize);
                        std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;
                    }
                }
            }
        }
    }

    if(Prank == 0){
        std::ofstream file(outputDir + "/cholesky-mpi.csv");
        if(!file.is_open()){
            std::cerr << "Cannot open file " << outputDir + "/cholesky-mpi.csv" << std::endl;
            return 1;
        }

    file << "NbGpu,MatrixSize,BlockSize,Multiprio,PrioPair,FavorLocality,MinDuration,MaxDuration,AvgDuration,TotalTransfer,MaxTransfer,DeviceToHostTransfer,HostToDeviceTransfer,DeviceToDeviceTransfer" << std::endl;
    int idxDuration = 0;
    for(auto useMultiprioAndPairs: schedPairConf){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                    const bool useMultiprio = std::get<0>(useMultiprioAndPairs);
                    const bool usePrioPairs = std::get<1>(useMultiprioAndPairs);
                    const bool useLocality = std::get<2>(useMultiprioAndPairs);

                    file << idxGpu << "," << MatrixSize << "," << BlockSize << "," 
                        << (useMultiprio?"TRUE":"FALSE") << ","
                        << (usePrioPairs?"TRUE":"FALSE") << ","
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


    return 0;
}
