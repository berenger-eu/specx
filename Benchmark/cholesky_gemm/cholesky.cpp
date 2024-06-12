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

auto choleskyFactorization(const int NbLoops, SpBlas::Block blocks[], const int inMatrixDim, const int inBlockDim,
                           const int nbGpu, const bool useMultiPrioScheduler){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

#ifdef SPECX_COMPILE_WITH_CUDA
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(SpUtils::DefaultNumThreads(), nbGpu), std::move(scheduler));
#else
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(), std::move(scheduler));
#endif

#ifdef SPECX_COMPILE_WITH_CUDA
    struct CudaHandles{
        cublasHandle_t blasHandle;
        cusolverDnHandle_t solverHandle;
        int solverBuffSize;
        double* solverBuffer;
        int* cuinfo;
    };
     std::vector<CudaHandles> handles(ce.getNbCudaWorkers());
     const int offsetWorker = ce.getNbCpuWorkers() + 1;
     ce.execOnWorkers([&handles, offsetWorker, &ce, inBlockDim, &blocks](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
             assert(offsetWorker <= id && id < offsetWorker + ce.getNbCudaWorkers());
             SpDebugPrint() << "Worker " << id << " will now initiate cublas...";
             auto& hdl = handles[id-offsetWorker];
             CUBLAS_ASSERT(cublasCreate(&hdl.blasHandle));
             CUBLAS_ASSERT(cublasSetStream(hdl.blasHandle, SpCudaUtils::GetCurrentStream()));
             CUSOLVER_ASSERT(cusolverDnCreate(&hdl.solverHandle));
             CUSOLVER_ASSERT(cusolverDnSetStream(hdl.solverHandle, SpCudaUtils::GetCurrentStream()));
             CUSOLVER_ASSERT(cusolverDnDpotrf_bufferSize(
                                 hdl.solverHandle,
                                 CUBLAS_FILL_MODE_LOWER,
                                 inBlockDim,
                                 blocks[0].values.get(),
                                 inBlockDim,
                                 &hdl.solverBuffSize));
             CUDA_ASSERT(cudaMalloc((void**)&hdl.solverBuffer , sizeof(double)*hdl.solverBuffSize));
             CUDA_ASSERT(cudaMalloc((void**)&hdl.cuinfo , sizeof(int)));
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

        // Compute the blocks
        for(int k = 0 ; k < nbBlocks ; ++k){
            // TODO put for syrk and gemm ? const double rbeta = (j==0) ? beta : 1.0;
            // POTRF( RW A(k,k) )
            tg.task(SpPriority(0), SpWrite(blocks[k*nbBlocks+k]),
                SpCpu([inBlockDim](SpBlas::Block& block){
                    SpBlas::potrf( SpBlas::FillMode::LWPR, inBlockDim, block.values.get(), inBlockDim );
                })
    #ifdef SPECX_COMPILE_WITH_CUDA
             , SpCuda([inBlockDim, offsetWorker, &handles](SpDeviceDataView<SpBlas::Block> param) {
                const int idxCudaWorker = SpUtils::GetThreadId() - offsetWorker;
                assert(idxCudaWorker < int(handles.size()));
                CUSOLVER_ASSERT(cusolverDnDpotrf(
                    handles[idxCudaWorker].solverHandle,
                    CUBLAS_FILL_MODE_LOWER,
                    inBlockDim,
                    (double*)param.getRawPtr(),
                    inBlockDim,
                    handles[idxCudaWorker].solverBuffer,
                    handles[idxCudaWorker].solverBuffSize,
                    handles[idxCudaWorker].cuinfo));
    #ifndef NDEBUG
                int info;
                CUDA_ASSERT(cudaMemcpy(&info, handles[idxCudaWorker].cuinfo, sizeof(int), cudaMemcpyDeviceToHost));
                assert(info >= 0);
    #endif
            })
    #endif
                    ).setTaskName(std::string("potrf -- (W-")+std::to_string(k)+","+std::to_string(k)+")");

            for(int m = k + 1 ; m < nbBlocks ; ++m){
                // TRSM( R A(k,k), RW A(m, k) )
                tg.task(SpPriority(1), SpRead(blocks[k*nbBlocks+k]), SpWrite(blocks[k*nbBlocks+m]),
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
                    const double one = 1;
                    CUBLAS_ASSERT( cublasDtrsm( handles[idxCudaWorker].blasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                                                CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                            inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                            (double*)paramB.getRawPtr(), inBlockDim));
                })
        #endif
                        ).setTaskName(std::string("TRSM -- (R-")+std::to_string(k)+","+std::to_string(k)+") (W-"+std::to_string(m)+","+std::to_string(k)+")");
            }

            for(int n = k+1 ; n < nbBlocks ; ++n){
                // SYRK( R A(n,k), RW A(n, n) )
                tg.task(SpPriority(1), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[n*nbBlocks+n]),
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
                    const double one = 1;
                    const double minusone = -1;
                    CUBLAS_ASSERT( cublasDsyrk( handles[idxCudaWorker].blasHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                            inBlockDim, inBlockDim, &minusone, (const double*)paramA.getRawPtr(), inBlockDim,
                            &one, (double*)paramC.getRawPtr(), inBlockDim ) );
                })
        #endif
                        ).setTaskName(std::string("SYRK -- (R-")+std::to_string(n)+","+std::to_string(k)+") (W-"+std::to_string(n)+","+std::to_string(n)+")");

                for(int m = k+1 ; m < nbBlocks ; ++m){
                    // GEMM( R A(m, k), R A(n, k), RW A(m, n))
                    tg.task(SpPriority(3), SpRead(blocks[k*nbBlocks+m]), SpRead(blocks[k*nbBlocks+n]), SpWrite(blocks[m*nbBlocks+n]),
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
                        const double one = 1;
                        const double minusone = -1;
                        CUBLAS_ASSERT( cublasDgemm( handles[idxCudaWorker].blasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                inBlockDim, inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                (const double*)paramB.getRawPtr(), inBlockDim,
                                &minusone, (double*)paramC.getRawPtr(), inBlockDim ) );
                    })
            #endif
                            ).setTaskName(std::string("GEMM -- (R-")+std::to_string(m)+","+std::to_string(k)+")(R-"+std::to_string(n)+","+std::to_string(k)+")(W-"+std::to_string(m)+","+std::to_string(n)+")");
                }
            }
        }
        tg.waitAllTasks();
        timer.stop();

#ifdef SPECX_COMPILE_WITH_CUDA
        for(int i = 0 ; i < nbBlocks ; ++i){
            for(int j = 0 ; j < nbBlocks ; ++j){
                tg.syncDataOnCpu(blocks[i*nbBlocks+j]);
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
            CUBLAS_ASSERT(cublasDestroy(hdl.blasHandle));
            CUSOLVER_ASSERT(cusolverDnDestroy(hdl.solverHandle));
            CUDA_ASSERT(cudaFree(hdl.solverBuffer));
        }
     });
#endif

    ce.stopIfNotAlreadyStopped();

    minMaxAvg[2] /= NbLoops;
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
                    const auto minMaxAvg = choleskyFactorization(NbLoops, blocks.get(), MatrixSize, BlockSize, 
                                                                idxGpu, useMultiprio);
                    allDurations.push_back(minMaxAvg);
                    std::cout << "     - Duration = " << minMaxAvg[0] << " " << minMaxAvg[1] << " " << minMaxAvg[2] << std::endl;
                    if(printValues){
                        std::cout << "Blocks after facto:\n";
                        SpBlas::printBlocks(blocks.get(), MatrixSize, BlockSize);
                    }
                    /////////////////////////////////////////////////////////
                    choleskyFactorizationMatrix(NbLoops, matrix.get(), MatrixSize);
                    if(printValues){
                        std::cout << "Matrix after facto:\n";
                        SpBlas::printMatrix(matrix.get(), MatrixSize);
                    }
                    /////////////////////////////////////////////////////////
                    const double errorAfterFacto = SpBlas::diffMatrixBlocks(matrix.get(), blocks.get(), MatrixSize, BlockSize);
                    std::cout << "Accuracy after facto : " << errorAfterFacto << std::endl;
                }
            }
        }
    }

    std::ofstream file(outputDir + "/gemm.csv");
    if(!file.is_open()){
        std::cerr << "Cannot open file " << outputDir + "/gemm.csv" << std::endl;
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


    return 0;
}
