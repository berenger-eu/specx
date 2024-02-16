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
        out[i] = a*x[i] + y[i];
}
#endif


void BenchmarkTest(int argc, char** argv){
    CLsimple args("Axpy", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int size = 100;
    args.addParameter<int>({"sz" ,"size"}, "Size", size, 1024);

    int nbthreads;
    args.addParameter<int>({"th"}, "nbthreads", nbthreads, 256);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return;
    }

    Vector<float> x;
    x.data.resize(size, 1);
    Vector<float> y;
    y.data.resize(size, 1);
    Vector<float> z;
    z.data.resize(size, 0);
    const float a = 2;

#ifdef SPECX_COMPILE_WITH_CUDA
    SpCudaUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers());
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif
    SpTaskGraph tg;

    tg.computeOn(ce);

    tg.task(SpCommutativeWrite(z),SpRead(x),SpRead(y),
#ifndef SPECX_COMPILE_WITH_CUDA
            SpCpu([ta=a](Vector<float>& tz, const Vector<float>& tx, const Vector<float>& ty) {
                for(int idx = 0 ; idx < int(tz.data.size()) ; ++idx){
                    tz.data[idx] = ta*tx.data[idx]*ty.data[idx];
                }
            })
#else
            SpCuda([a, nbthreads](SpDeviceDataView<Vector<float>> paramZ,
                       const SpDeviceDataView<const Vector<float>> paramX,
                       const SpDeviceDataView<const Vector<float>> paramY) {
                const int size = paramZ.data().getSize();
                const int nbBlocks = (size + nbthreads-1)/nbthreads;
                cu_axpy<float><<<nbBlocks, nbthreads,0,SpCudaUtils::GetCurrentStream()>>>
                    (size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
            })
#endif
            );

#ifdef SPECX_COMPILE_WITH_CUDA
    tg.task(SpWrite(z),
    SpCpu([](Vector<float>&) {
    })
    );
#endif

    tg.waitAllTasks();

    std::cout << "Generate trace ./axpy-simu.svg" << std::endl;
    tg.generateTrace("./axpy-simu.svg", false);
}


int main(int argc, char** argv){
    BenchmarkTest(argc, argv);

    return 0;
}
