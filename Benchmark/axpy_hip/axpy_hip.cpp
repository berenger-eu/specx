#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

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

#include "hip/hip_runtime.h"
#include <hipblas.h>
#include <hiprand.h>
 

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif



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

#ifdef SPECX_COMPILE_WITH_HIP
template <class NumType>
__global__ void hip_axpy(int n, NumType a,NumType b, NumType *x, NumType *y, NumType *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a*x[i] + b*y[i];
}
#endif


void BenchmarkTest(int argc, char** argv){
    // ./axpy_hip --sz=50 --th=256
    std::cout<<"[INFO]: BENCHMARK Vr=a*Va+b*Vb"<<"\n";

    // BEGIN:: Init size vectors and Nb Threads in GPU HIP AMD
    CLsimple args("Axpy", argc, argv);
    args.addParameterNoArg({"help"}, "help");
    int size = 100;
    args.addParameter<int>({"sz" ,"size"}, "Size", size, 1024);
    int nbthreads;
    args.addParameter<int>({"th"}, "nbthreads", nbthreads, 256);
    args.parse();
    if(!args.isValid() || args.hasKey("help")){
      args.printHelp(std::cout);
      return;
    }
    // END:: Init size vectors and Nb Threads in GPU HIP AMD

    // BEGIN:: Init part
    Vector<float> x; x.data.resize(size, 3);
    Vector<float> y; y.data.resize(size, 1);
    Vector<float> z; z.data.resize(size, 0);
    const float coeff_a = 2;
    const float coeff_b = 3;
    // END:: Init part

    #ifdef SPECX_COMPILE_WITH_HIP
    std::cout<<"[INFO]: Start Hip Part..."<<"\n";
    // BEGIN:: Task definition
    //SpHipUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers());
    static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,"should be stdvec");

    SpTaskGraph tg;
    tg.computeOn(ce);
    tg.task(SpCommutativeWrite(z),SpRead(x),SpRead(y),
            SpHip([coeff_a,coeff_b, nbthreads](SpDeviceDataView<Vector<float>> paramZ,
                       const SpDeviceDataView<const Vector<float>> paramX,
                       const SpDeviceDataView<const Vector<float>> paramY) {
                            const int size = paramX.getRawSize()/sizeof(float);
                            //std::cout<<"NbElements="<<size<<"\n";
                            const int nbBlocks = (size + nbthreads-1)/nbthreads;
                            hipLaunchKernelGGL(hip_axpy<float>,nbBlocks, nbthreads,0,SpHipUtils::GetCurrentStream(),
                                size, 
                                coeff_a,coeff_b,(float*)paramX.getRawPtr(),(float*)paramY.getRawPtr(),(float*)paramZ.getRawPtr()
                            );

            })

    );

    tg.task(SpWrite(z), SpCpu([](Vector<float>&) { }));
    // END:: Task definition

    // BEGIN:: Task execution
    tg.waitAllTasks();
    //tg.stopIfNotAlreadyStopped();
    // END:: Task execution

    // BEGIN:: Show results
    std::cout<<"[INFO]: Generate trace..."<<"\n";
    tg.generateTrace("./axpy-simu-hip.svg",true);

    std::cout<<"[INFO]: Results..."<<"\n";
    std::cout<<"[INFO]: a         = "<<coeff_a<<"\n";
    std::cout<<"[INFO]: b         = "<<coeff_b<<"\n";
    std::cout<<"[INFO]: Vector   x= "; for(int k=0;k<size;k++) { std::cout<<x.data[k]; } std::cout<<"\n";
    std::cout<<"[INFO]: Vector   y= "; for(int k=0;k<size;k++) { std::cout<<y.data[k]; } std::cout<<"\n";
    std::cout<<"[INFO]: z=a*x+b*y = "; for(int k=0;k<size;k++) { std::cout<<z.data[k]; } std::cout<<"\n";
    std::cout<<"\n";
    // END:: End results
    #endif
}


int main(int argc, char** argv){
    std::cout<<"\n";
    BenchmarkTest(argc, argv);
    std::cout<<"[INFO]: FINISHED..."<<"\n";
    return 0;
}
