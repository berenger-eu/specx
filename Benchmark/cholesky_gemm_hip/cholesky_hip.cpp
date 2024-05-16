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

#include "CholeskyFunctionsWrapper_hip.hpp"


//////////////////////////////////////////////////////////////////////////////

void choleskyFactorizationMatrix(const int NbLoops, double matrix[], const int inMatrixDim){
    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpBlas::potrf( SpBlas::FillMode::LWPR, inMatrixDim, matrix, inMatrixDim );
    }
}

void choleskyFactorization(const int NbLoops, SpBlas::Block blocks[], const int inMatrixDim, const int inBlockDim){
    const int nbBlocks = (inMatrixDim+inBlockDim-1)/inBlockDim;

    SpRuntime<> runtime;

#ifdef SPECX_COMPILE_WITH_HIP
    struct HipHandles{
        hipblasHandle_t blasHandle;
        hipsolverDnHandle_t solverHandle;
        int solverBuffSize;
        double* solverBuffer;
        int* cuinfo;
    };
     std::vector<HipHandles> handles(runtime.getNbHipWorkers());
     const int offsetWorker = runtime.getNbCpuWorkers() + 1;
     runtime.execOnWorkers([&handles, offsetWorker, &runtime, inBlockDim, &blocks](auto id, auto type){
         assert(id == SpUtils::GetThreadId());
         assert(type == SpUtils::GetThreadType());
         if(type == SpWorkerTypes::Type::HIP_WORKER){
             assert(offsetWorker <= id && id < offsetWorker + runtime.getNbHipWorkers());
             SpDebugPrint() << "Worker " << id << " will now initiate HIPBLAS...";
             auto& hdl = handles[id-offsetWorker];
             HIPBLAS_ASSERT(hipblasCreate(&hdl.blasHandle));
             HIPBLAS_ASSERT(hipblasSetStream(hdl.blasHandle, SpHipUtils::GetCurrentStream()));
             HIPSOLVER_ASSERT(hipsolverDnCreate(&hdl.solverHandle));
             HIPSOLVER_ASSERT(hipsolverDnSetStream(hdl.solverHandle, SpHipUtils::GetCurrentStream()));
             HIPSOLVER_ASSERT(hipsolverDnDpotrf_bufferSize (
                                 hdl.solverHandle,
                                 HIPBLAS_FILL_MODE_LOWER,
                                 inBlockDim,
                                 blocks[0].values.get(),
                                 inBlockDim,
                                 &hdl.solverBuffSize));
             HIP_ASSERT(hipMalloc((void**)&hdl.solverBuffer , sizeof(double)*hdl.solverBuffSize));
             HIP_ASSERT(hipMalloc((void**)&hdl.cuinfo , sizeof(int)));
         }
     });
#endif

     for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        // Compute the blocks
        for(int k = 0 ; k < nbBlocks ; ++k){
            // TODO put for syrk and gemm ? const double rbeta = (j==0) ? beta : 1.0;
            // POTRF( RW A(k,k) )
            runtime.task(SpPriority(0), SpWrite(blocks[k*nbBlocks+k]),
                SpCpu([inBlockDim](SpBlas::Block& block){
                    SpBlas::potrf( SpBlas::FillMode::LWPR, inBlockDim, block.values.get(), inBlockDim );
                })
    #ifdef SPECX_COMPILE_WITH_HIP
             , SpHip([inBlockDim, offsetWorker, &handles](SpDeviceDataView<SpBlas::Block> param) {
                const int idxHipWorker = SpUtils::GetThreadId() - offsetWorker;
                assert(idxHipWorker < int(handles.size()));
                HIPSOLVER_ASSERT(hipsolverDnDpotrf(
                    handles[idxHipWorker].solverHandle,
                    HIPBLAS_FILL_MODE_LOWER,
                    inBlockDim,
                    (double*)param.getRawPtr(),
                    inBlockDim,
                    handles[idxHipWorker].solverBuffer,
                    handles[idxHipWorker].solverBuffSize,
                    handles[idxHipWorker].cuinfo));
    #ifndef NDEBUG
                int info;
                HIP_ASSERT(hipMemcpy(&info, handles[idxHipWorker].cuinfo, sizeof(int), hipMemcpyDeviceToHost));
                assert(info >= 0);
    #endif
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
        #ifdef SPECX_COMPILE_WITH_HIP
              , SpHip([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                                  SpDeviceDataView<SpBlas::Block> paramB) {
                    const int idxHipWorker = SpUtils::GetThreadId() - offsetWorker;
                    assert(idxHipWorker < int(handles.size()));
                    const double one = 1;
                    HIPBLAS_ASSERT( hipblasDtrsm( handles[idxHipWorker].blasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER,
                                                HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT,
                            inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
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
        #ifdef SPECX_COMPILE_WITH_HIP
              , SpHip([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                                  SpDeviceDataView<SpBlas::Block> paramC) {
                    // paramA.getRawPtr(), paramA.getRawSize()
                    const int idxHipWorker = SpUtils::GetThreadId() - offsetWorker;
                    assert(idxHipWorker < int(handles.size()));
                    const double one = 1;
                    const double minusone = -1;
                    HIPBLAS_ASSERT( hipblasDsyrk( handles[idxHipWorker].blasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N,
                            inBlockDim, inBlockDim, &minusone, (const double*)paramA.getRawPtr(), inBlockDim,
                            &one, (double*)paramC.getRawPtr(), inBlockDim ) );
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
            #ifdef SPECX_COMPILE_WITH_HIP
                  , SpHip([inBlockDim, offsetWorker, &handles](const SpDeviceDataView<const SpBlas::Block> paramA,
                                      const SpDeviceDataView<const SpBlas::Block> paramB, SpDeviceDataView<SpBlas::Block> paramC) {
                        const int idxHipWorker = SpUtils::GetThreadId() - offsetWorker;
                        assert(idxHipWorker < int(handles.size()));
                        const double one = 1;
                        const double minusone = -1;
                        HIPBLAS_ASSERT( hipblasDgemm( handles[idxHipWorker].blasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T,
                                inBlockDim, inBlockDim, inBlockDim, &one, (const double*)paramA.getRawPtr(), inBlockDim,
                                (const double*)paramB.getRawPtr(), inBlockDim,
                                &minusone, (double*)paramC.getRawPtr(), inBlockDim ) );
                    })
            #endif
                            ).setTaskName(std::string("GEMM -- (R-")+std::to_string(m)+","+std::to_string(k)+")(R-"+std::to_string(n)+","+std::to_string(k)+")(W-"+std::to_string(m)+","+std::to_string(n)+")");
                }
            }
        }
     }

#ifdef SPECX_COMPILE_WITH_HIP
    for(int i = 0 ; i < nbBlocks ; ++i){
        for(int j = 0 ; j < nbBlocks ; ++j){
            runtime.syncDataOnCpu(blocks[i*nbBlocks+j]);
        }
    }
#endif

    runtime.waitAllTasks();

#ifdef SPECX_COMPILE_WITH_HIP
    runtime.execOnWorkers([&handles, offsetWorker](auto id, auto type){
        if(type == SpWorkerTypes::Type::HIP_WORKER){
            auto& hdl = handles[id-offsetWorker];
            HIPBLAS_ASSERT(hipblasDestroy(hdl.blasHandle));
            HIPSOLVER_ASSERT(hipsolverDnDestroy(hdl.solverHandle));
            HIP_ASSERT(hipFree(hdl.solverBuffer));
        }
     });
#endif

    runtime.stopAllThreads();
    runtime.generateDot("/tmp/graph.dot");
}

int main(int argc, char** argv){
    CLsimple args("Cholesky", argc, argv);

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
    choleskyFactorization(NbLoops, blocks.get(), MatrixSize, BlockSize);
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

    return 0;
}
