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

#ifdef SPECX_COMPILE_WITH_CUDA
thread_local cublasHandle_t handle;
#endif

auto gemm(const int NbLoops, SpBlas::Block blocksC[], const SpBlas::Block blocksA[], const SpBlas::Block blocksB[],
                           const int inMatrixDim, const int inBlockDim,
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
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif

#ifdef SPECX_COMPILE_WITH_CUDA
     ce.execOnWorkers([&ce](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::CUDA_WORKER){
             SpDebugPrint() << "Worker " << id << " will now initiate cublas...";
            CUBLAS_ASSERT(cublasCreate(&handle));
            CUBLAS_ASSERT(cublasSetStream(handle, SpCudaUtils::GetCurrentStream()));
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
        for(int i = 0 ; i < nbBlocks ; ++i){
            for(int j = 0 ; j < nbBlocks ; ++j){
                for(int k = 0 ; k < nbBlocks ; ++k){
                    tg.task(SpPriority(1), SpCommutativeWrite(blocksC[i*nbBlocks+j]),
                            SpRead(blocksA[k*nbBlocks+j]), SpRead(blocksB[i*nbBlocks+k]),
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
                    ).setTaskName(std::string("GEMM -- (")+std::to_string(i)+","+std::to_string(j)+")");
                }
            }
        }
    

        tg.waitAllTasks();
        timer.stop();

#ifdef SPECX_COMPILE_WITH_CUDA
        for(int i = 0 ; i < nbBlocks ; ++i){
            for(int j = 0 ; j < nbBlocks ; ++j){
                tg.task(SpRead(blocksC[i*nbBlocks+j]),
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

    for(bool useMultiprio: std::vector<bool>{true, false}){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            for(int BlockSize = MinBlockSize ; BlockSize <= MaxBlockSize ; BlockSize *= 2){
                for(int MatrixSize = MinMatrixSize ; MatrixSize <= MaxMatrixSize ; MatrixSize *= 2){
                    std::cout << "NbGpu = " << idxGpu << " MatrixSize = " << MatrixSize 
                        << " BlockSize = " << BlockSize << " Multiprio = " << useMultiprio << std::endl;

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
                    const auto minMaxAvg = gemm(NbLoops, blocksC.get(), blocksA.get(), blocksB.get(), 
                                                MatrixSize, BlockSize, idxGpu, useMultiprio);
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
                    if(checkAccuracy){
                        const double errorBeforeFacto = SpBlas::diffMatrixBlocks(matrixC.get(), blocksC.get(), MatrixSize, BlockSize);
                        std::cout << "Accuracy before facto : " << errorBeforeFacto << std::endl;
                    }
                }
            }
        }
    }

    // Print out csv
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
