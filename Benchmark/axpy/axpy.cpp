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


#ifdef SPECX_COMPILE_WITH_CUDA
template <class NumType>
__global__ void cu_axpy(int n, NumType a, NumType *x, NumType *y, NumType *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < n)
        out[i] += a*x[i] + y[i];
}
#endif


double BenchmarkTest(const int NbLoops, const int nbGpu, const int nbblocks, const int blocksize, const int cudanbthreads){   
    std::vector<Vector<float>> x(nbblocks, Vector<float>(blocksize, 1));
    std::vector<Vector<float>> y(nbblocks, Vector<float>(blocksize, 1));
    std::vector<Vector<float>> z(nbblocks, Vector<float>(blocksize, 0));
    const float a = 2;

#ifdef SPECX_COMPILE_WITH_CUDA
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(SpUtils::DefaultNumThreads(), nbGpu));
#elif defined(SPECX_COMPILE_WITH_HIP)
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(SpUtils::DefaultNumThreads(), nbGpu));
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif
    SpTaskGraph tg;

    tg.computeOn(ce);

    SpTimer timer;

    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        for(int idxBlock = 0 ; idxBlock < nbblocks ; ++idxBlock){
            tg.task(SpCommutativeWrite(z[idxBlock]),SpRead(x[idxBlock]),SpRead(y[idxBlock]),
                SpCpu([ta=a](Vector<float>& tz, const Vector<float>& tx, const Vector<float>& ty) {
                    for(int idx = 0 ; idx < int(tz.data.size()) ; ++idx){
                        tz.data[idx] = ta*tx.data[idx]*ty.data[idx];
                    }
                })
#ifdef SPECX_COMPILE_WITH_CUDA
                , SpCuda([a, cudanbthreads](SpDeviceDataView<Vector<float>> paramZ,
                        const SpDeviceDataView<const Vector<float>> paramX,
                        const SpDeviceDataView<const Vector<float>> paramY) {
                    const int size = paramZ.data().getSize();
                    const int cudanbblocks = (size + cudanbthreads-1)/cudanbthreads;
                    cu_axpy<float><<<cudanbblocks, cudanbthreads,0,SpCudaUtils::GetCurrentStream()>>>
                        (size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
                })
#endif
#ifdef SPECX_COMPILE_WITH_Hip
                , SpHip([a, cudanbthreads](SpDeviceDataView<Vector<float>> paramZ,
                        const SpDeviceDataView<const Vector<float>> paramX,
                        const SpDeviceDataView<const Vector<float>> paramY) {
                    const int size = paramZ.data().getSize();
                    const int cudanbblocks = (size + cudanbthreads-1)/cudanbthreads;
                    cu_axpy<float><<<cudanbblocks, cudanbthreads,0,SpCudaUtils::GetCurrentStream()>>>
                        (size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
                })
#endif
            );
        }
    }

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
    for(int idxBlock = 0 ; idxBlock < nbblocks ; ++idxBlock){
        tg.task(SpWrite(z[idxBlock]),
        SpCpu([](Vector<float>&) {
        })
        );
    }
#endif

    tg.waitAllTasks();

    timer.stop();
    return timer.getElapsed();
}


int main(int argc, char** argv){
    CLsimple args("Axpy", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int NbLoops = 100;
    args.addParameter<int>({"lp" ,"nbloops"}, "NbLoops", NbLoops, NbLoops);

    int minnbblocks = 10;
    args.addParameter<int>({"minnbb" ,"minnbblocks"}, "Min NbBlocks", minnbblocks, minnbblocks);

    int maxnbblocks = 100;
    args.addParameter<int>({"maxnbb" ,"maxnbblocks"}, "Max NbBlocks", maxnbblocks, maxnbblocks);

    int minblocksize = 512;
    args.addParameter<int>({"minbs" ,"minblocksize"}, "Min Block size", minblocksize, minblocksize);

    int maxblocksize = 512;
    args.addParameter<int>({"maxbs" ,"maxblocksize"}, "Max Block size", maxblocksize, maxblocksize);

    int cudanbthreads = 256;
    args.addParameter<int>({"cuth"}, "cuthreads", cudanbthreads, cudanbthreads);

    std::string outputDir = "./";
    args.addParameter<std::string>({"od"}, "outputdir", outputDir, outputDir);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return;
    }

    const int nbGpus = SpCudaUtils::GetNbDevices();

    std::vector<double> allDurations;

    for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
        if(idxGpu > 0){
#ifdef SPECX_COMPILE_WITH_CUDA            
            std::cout << ">> GPU " << idxGpu << " : " << SpCudaUtils::GetDeviceName(idxGpu-1) << std::endl;
#elif defined(SPECX_COMPILE_WITH_HIP)
            std::cout << ">> GPU " << idxGpu << " : " << SpHipUtils::GetDeviceName(idxGpu-1) << std::endl;
#endif                        
        }
        else{
            std::cout << ">> CPU " << SpUtils::DefaultNumThreads() << std::endl;
        }
        for(int idxBlock = minnbblocks ; idxBlock <= maxnbblocks ; idxBlock += 10){
            for(int idxSize = minblocksize ; idxSize <= maxblocksize ; idxSize *= 2){
                std::cout << "  - NbBlocks = " << idxBlock << " BlockSize = " << idxSize << std::endl;
                const double duration = BenchmarkTest(NbLoops, idxGpu, idxBlock, idxSize, cudanbthreads);
                std::cout << "     - Duration = " << duration << std::endl;
                std::cout << "     - End" << std::endl;
                allDurations.push_back(duration);
            }
        }
    }

    // Print out csv
    std::ofstream file(outputDir + "/axpy.csv");
    if(!file.is_open()){
        std::cerr << "Cannot open file " << outputDir + "/axpy.csv" << std::endl;
        return 1;
    }

    file << "NbGpu,NbBlocks,BlockSize,Duration" << std::endl;
    int idxDuration = 0;
    for(int idxGpu = 0 ; idxGpu <= nbGpus ; ++idxGpu){
        for(int idxBlock = minnbblocks ; idxBlock <= maxnbblocks ; idxBlock += 10){
            for(int idxSize = minblocksize ; idxSize <= maxblocksize ; idxSize *= 2){
                file << idxGpu << "," << idxBlock << "," << idxSize << "," << allDurations[idxDuration++] << std::endl;
            }
        }
    }

    return 0;
}
