///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"

#include "hip/hip_runtime.h"

__global__ void inc_var(int* ptr, int size){
    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
        ptr[idx]++;
    }
}


#define DIM_DATA 10

struct MemmovClassExample {
    int data[DIM_DATA];


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
        return DIM_DATA*sizeof(int);
    }

    template <class DeviceMemmov>
    auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == DIM_DATA*sizeof(int));
        mover.copyHostToDevice(reinterpret_cast<int*>(devicePtr), &data[0],sizeof(int)*DIM_DATA);
        return DataDescr(DIM_DATA);
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size, const DataDescr& /*inDataDescr*/){
        assert(size == DIM_DATA*sizeof(int));
        mover.copyDeviceToHost(&data[0], reinterpret_cast<int*>(devicePtr),sizeof(int));
    }
};





class SimpleGpuTest : public UTester< SimpleGpuTest > {
    using Parent = UTester< SimpleGpuTest >;
    void Test(){
        //SpHipUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(1,1,2));
        SpTaskGraph tg;
        int a = 0;
        int b = 0;

        tg.computeOn(ce);

        tg.task(SpRead(a), SpRead(b),
                    SpHip([]([[maybe_unused]] SpDeviceDataView<const int> paramA,
                             [[maybe_unused]] SpDeviceDataView<const int> paramB) {
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(a),
                    SpHip([](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                        hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                            paramA.objPtr(), 1);
            #else
                        (*paramA.objPtr())++;
            #endif
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(b),
                    SpHip([](SpDeviceDataView<int> paramB) {
            #ifndef SPECX_EMUL_GPU
                        hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                            paramB.objPtr(), 1);
            #else
                        (*paramB.objPtr())++;
            #endif
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpRead(a), SpWrite(b),
                    SpCpu([](const int& paramA, int& paramB) {
                        paramB = paramA + paramB;
                    })
        );


        tg.task(SpWrite(a),
                    SpCpu([](int& paramA) {
                        paramA++;
                    }),
                    SpHip(
                        [](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                        hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                            paramA.objPtr(), 1);
            #else
                        (*paramA.objPtr())++;
            #endif
                    })
        );

        tg.task(SpWrite(a), SpWrite(b),
                    SpCpu([](int& paramA, int& paramB) {
                        paramA++;
                        paramB++;
                    })
        );

        tg.waitAllTasks();

        std::cout<<"a="<<a<< "\n";
        std::cout<<"b="<<b<< "\n";
        std::cout<<"\n";


        UASSERTETRUE(a == 3);
        UASSERTETRUE(b == 3);
    }


    void TestVec(){
        //SpHipUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(1,1,2));
        SpTaskGraph tg;
        std::vector<int> a(100,0);
        std::vector<int> b(100,0);

        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,
                      "should be stdvec");

        tg.computeOn(ce);

        tg.task(SpRead(a),
            SpHip([]([[maybe_unused]] SpDeviceDataView<const std::vector<int>> paramA) {
            })
        );

        tg.task(SpWrite(a),
            SpHip([](SpDeviceDataView<std::vector<int>> paramA) {
                hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                                   paramA.array(),paramA.nbElements());
                std::this_thread::sleep_for(std::chrono::seconds(2));
            })
        );

        tg.task(SpWrite(b),
            SpHip([](SpDeviceDataView<std::vector<int>> paramB) {
                hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                                   paramB.array(),paramB.nbElements());
            })
        );

        tg.task(SpRead(a), SpWrite(b),
            SpCpu([](const std::vector<int>& paramA, std::vector<int>& paramB) {
                assert(paramA.size() == paramB.size());
                for(int idx = 0 ; idx < int(paramA.size()) ; ++idx){
                    paramB[idx] = paramA[idx] + paramB[idx];
                }
            })
        );

        tg.task(SpWrite(a),
            SpCpu([](std::vector<int>& paramA) {
                for(auto& va : paramA){
                    va++;
                }
            }),
            SpHip([](SpDeviceDataView<std::vector<int>> paramA) {
                hipLaunchKernelGGL(inc_var,1,1,0,SpHipUtils::GetCurrentStream(),
                        paramA.array(),paramA.nbElements());

            })
        );

        tg.task(SpWrite(a), SpWrite(b),
            SpCpu([](std::vector<int>& paramA, std::vector<int>& paramB) {
                for(auto& va : paramA){
                    va++;
                }
                for(auto& vb : paramB){
                    vb++;
                }
            })
        );

        tg.waitAllTasks();

        for(auto& va : a){
            UASSERTETRUE(va == 3);
        }
        for(auto& vb : b){
            UASSERTETRUE(vb == 3);
        }
    }




    void TestMemMove(){
        //SpHipUtils::PrintInfo();
        MemmovClassExample obj;
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(1,1,2));
        static_assert(SpDeviceDataView<MemmovClassExample>::MoveType == SpDeviceDataUtils::DeviceMovableType::MEMMOV,"should be memmov");
        SpTaskGraph tg;
        tg.computeOn(ce);
        tg.task(SpRead(obj),
            SpHip([]([[maybe_unused]] SpDeviceDataView<const MemmovClassExample> objv) {
            })
        );

        tg.task(SpWrite(obj),
            SpHip([](SpDeviceDataView<MemmovClassExample> objv) {
            })
        );
        tg.waitAllTasks();
    }


    void SetTests() {
        //Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
        Parent::AddTest(&SimpleGpuTest::TestVec, "Basic gpu test with vec");
        Parent::AddTest(&SimpleGpuTest::TestMemMove, "Basic gpu test with memmov");
    }
};

// You must do this
TestClass(SimpleGpuTest)
