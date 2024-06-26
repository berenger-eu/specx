///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>
#include <iostream>

#include <clsimple.hpp>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"

#include "Scheduler/SpMultiPrioScheduler.hpp"


template <class NumType>
struct Vector{
    std::vector<NumType> data;

    Vector() = default;
    Vector(const std::size_t size, const NumType value) : data(size, value){}

    /////////////////////////////////////////////////////////////

    class DataDescr {
        std::size_t size;
    public:
        explicit DataDescr(const std::size_t inSize = 0) : size(inSize){}

        auto getSize() const{
            return size;
        }
    };

    using DataDescriptor = DataDescr;

    std::size_t memmovNeededSize() const{
        return sizeof(NumType)*data.size();
    }

    template <class DeviceMemmov>
    auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
        assert(size == sizeof(NumType)*data.size());
        NumType* doubleDevicePtr = reinterpret_cast<NumType*>(devicePtr);
        mover.copyHostToDevice(doubleDevicePtr, data.data(), sizeof(NumType)*data.size());
        return DataDescr(data.size());
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
        assert(size == sizeof(NumType)*data.size());
        NumType* doubleDevicePtr = reinterpret_cast<NumType*>(devicePtr);
        mover.copyDeviceToHost(data.data(), doubleDevicePtr, sizeof(NumType)*data.size());
    }
};


#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
template <class NumType>
__global__ void cu_axpy(int n, NumType a, NumType *x, NumType *y, NumType *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n)
        out[i] += a*x[i] + y[i];
}
#endif


auto BenchmarkTest(const int NbLoops, const int nbGpu, const int nbblocks, const int blocksize, const int gpunbthreads,
                    const bool useMultiPrioScheduler){   
    std::vector<Vector<float>> x(nbblocks, Vector<float>(blocksize, 1));
    std::vector<Vector<float>> y(nbblocks, Vector<float>(blocksize, 1));
    std::vector<Vector<float>> z(nbblocks, Vector<float>(blocksize, 0));
    const float a = 2;

#ifdef SPECX_COMPILE_WITH_CUDA
    std::unique_ptr<SpAbstractScheduler> scheduler;
    if(useMultiPrioScheduler == false){
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpHeterogeneousPrioScheduler());
    }
    else{
        scheduler = std::unique_ptr<SpAbstractScheduler>(new SpMultiPrioScheduler());
    }
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuGpuWorkers(SpUtils::DefaultNumThreads(), nbGpu), std::move(scheduler));
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif

    std::vector<double> minMaxAvg(3);
    minMaxAvg[0] = std::numeric_limits<double>::max();
    minMaxAvg[1] = std::numeric_limits<double>::min();
    minMaxAvg[2] = 0;

    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        SpTaskGraph tg;

        tg.computeOn(ce);

        SpTimer timer;

        for(int idxBlock = 0 ; idxBlock < nbblocks ; ++idxBlock){
            tg.task(SpCommutativeWrite(z[idxBlock]),SpRead(x[idxBlock]),SpRead(y[idxBlock]),
                SpCpu([ta=a](Vector<float>& tz, const Vector<float>& tx, const Vector<float>& ty) {
                    for(int idx = 0 ; idx < int(tz.data.size()) ; ++idx){
                        tz.data[idx] = ta*tx.data[idx]*ty.data[idx];
                    }
                })
#ifdef SPECX_COMPILE_WITH_CUDA
                , SpCuda([a, gpunbthreads](SpDeviceDataView<Vector<float>> paramZ,
                        const SpDeviceDataView<const Vector<float>> paramX,
                        const SpDeviceDataView<const Vector<float>> paramY) {
                    const int size = paramZ.data().getSize();
                    const int gpunbblocks = (size + gpunbthreads-1)/gpunbthreads;
                    cu_axpy<float><<<gpunbblocks, gpunbthreads,0,SpCudaUtils::GetCurrentStream()>>>
                        (size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
                })
#endif
#ifdef SPECX_COMPILE_WITH_HIP
                , SpHip([a, gpunbthreads](SpDeviceDataView<Vector<float>> paramZ,
                        const SpDeviceDataView<const Vector<float>> paramX,
                        const SpDeviceDataView<const Vector<float>> paramY) {
                    const int size = paramZ.data().getSize();
                    const int gpunbblocks = (size + gpunbthreads-1)/gpunbthreads;
                    hipLaunchKernelGGL( cu_axpy<float>, gpunbblocks, gpunbthreads,0,SpHipUtils::GetCurrentStream(),
                        size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
                })
#endif
            );
        }

        tg.waitAllTasks();
        timer.stop();

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
        for(int idxBlock = 0 ; idxBlock < nbblocks ; ++idxBlock){
            tg.task(SpWrite(z[idxBlock]),
            SpCpu([](Vector<float>&) {
            })
            );
        }
        tg.waitAllTasks();
#endif

        minMaxAvg[0] = std::min(minMaxAvg[0], timer.getElapsed());
        minMaxAvg[1] = std::max(minMaxAvg[1], timer.getElapsed());
        minMaxAvg[2] += timer.getElapsed();
    }

    minMaxAvg[2] /= NbLoops;

    return minMaxAvg;
}


int main(int argc, char** argv){
    CLsimple args("Axpy", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int NbLoops = 100;
    args.addParameter<int>({"lp" ,"nbloops"}, "NbLoops", NbLoops, NbLoops);

    int minnbblocks = 16;
    args.addParameter<int>({"minnbb" ,"minnbblocks"}, "Min NbBlocks", minnbblocks, minnbblocks);

    int maxnbblocks = 256;
    args.addParameter<int>({"maxnbb" ,"maxnbblocks"}, "Max NbBlocks", maxnbblocks, maxnbblocks);

    int minblocksize = 512;
    args.addParameter<int>({"minbs" ,"minblocksize"}, "Min Block size", minblocksize, minblocksize);

    int maxblocksize = 512;
    args.addParameter<int>({"maxbs" ,"maxblocksize"}, "Max Block size", maxblocksize, maxblocksize);

    int gpunbthreads = 256;
    args.addParameter<int>({"gputh"}, "gputhreads", gpunbthreads, gpunbthreads);

    std::string outputDir = "./";
    args.addParameter<std::string>({"od"}, "outputdir", outputDir, outputDir);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return 1;
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
    std::cout << "CPU number of cores " << SpUtils::DefaultNumThreads() << std::endl;

    std::vector<std::vector<double>> allDurations;

    for(bool useMultiprio: std::vector<bool>{true, false}){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            std::cout << " - Gpu = " << idxGpu << " Multiprio = " << (useMultiprio?"TRUE":"FALSE") << std::endl;
            for(int idxNbBlocks = minnbblocks ; idxNbBlocks <= maxnbblocks ; idxNbBlocks *= 2){
                for(int idxSize = minblocksize ; idxSize <= maxblocksize ; idxSize *= 2){
                    std::cout << "  - NbBlocks = " << idxNbBlocks << " BlockSize = " << idxSize << std::endl;
                    const auto minMaxAvg = BenchmarkTest(NbLoops, idxGpu, idxNbBlocks, idxSize, gpunbthreads, useMultiprio);
                    std::cout << "     - Duration = " << minMaxAvg[0] << " " << minMaxAvg[1] << " " << minMaxAvg[2] << std::endl;
                    std::cout << "     - End" << std::endl;
                    allDurations.push_back(minMaxAvg);
                }
            }
        }
    }

    // Print out csv
    std::ofstream file(outputDir + "/axpy.csv");
    if(!file.is_open()){
        std::cerr << "Cannot open file " << outputDir + "/axpy.csv" << std::endl;
        return 1;
    }

    file << "NbGpu,NbBlocks,BlockSize,Multiprio,MinDuration,MaxDuration,AvgDuration" << std::endl;
    int idxDuration = 0;
    for(bool useMultiprio: std::vector<bool>{true, false}){
        for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
            for(int idxNbBlocks = minnbblocks ; idxNbBlocks <= maxnbblocks ; idxNbBlocks *= 2){
                for(int idxSize = minblocksize ; idxSize <= maxblocksize ; idxSize *= 2){
                    file << idxGpu << "," << idxNbBlocks << "," << idxSize << "," 
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
