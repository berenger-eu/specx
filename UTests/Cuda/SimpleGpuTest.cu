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

__global__ void inc_var(int* ptr, int size){
    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
        ptr[idx]++;
    }
}

class MemmovClassExample{
    int data[10];
public:
    std::size_t memmovNeededSize() const{
        return 10*sizeof(int);
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10*sizeof(int));
        mover.copyHostToDevice(reinterpret_cast<int*>(devicePtr), &data[0], 10*sizeof(int));
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10*sizeof(int));
        mover.copyDeviceToHost(&data[0], reinterpret_cast<int*>(devicePtr), 10*sizeof(int));
    }

    struct View{
        View(){}
        View(void* devicePtr, std::size_t size){}
    };
    using DeviceDataType = View;
};

class SimpleGpuTest : public UTester< SimpleGpuTest > {
    using Parent = UTester< SimpleGpuTest >;

    void Test(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        int a = 0;
        int b = 0;

        tg.computeOn(ce);

        tg.task(SpRead(a),
                    SpCuda([]([[maybe_unused]] SpDeviceDataView<const int> paramA) {
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(a),
                    SpCuda([](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.objPtr(), 1);
            #else
                        (*paramA.objPtr())++;
            #endif
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(b),
                    SpCuda([](SpDeviceDataView<int> paramB) {
            #ifndef SPECX_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramB.objPtr(), 1);
            #else
                        (*paramB.objPtr())++;
            #endif
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
                    SpCuda(
                        [](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.objPtr(), 1);
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

        UASSERTETRUE(a == 3);
        UASSERTETRUE(b == 3);
    }

    void TestVec(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        std::vector<int> a(100,0);
        std::vector<int> b(100,0);

        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,
                      "should be stdvec");

        tg.computeOn(ce);

        tg.task(SpWrite(a),
            SpCuda([](SpDeviceDataView<std::vector<int>> paramA) {
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                                                                   paramA.nbElements());

                std::this_thread::sleep_for(std::chrono::seconds(2));
            })
        );

        tg.task(SpWrite(b),
            SpCuda([](SpDeviceDataView<std::vector<int>> paramB) {
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramB.array(),
                    paramB.nbElements());
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
            SpCuda([](SpDeviceDataView<std::vector<int>> paramA) {
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                                                                   paramA.nbElements());
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
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        tg.computeOn(ce);

        static_assert(SpDeviceDataView<MemmovClassExample>::MoveType == SpDeviceDataUtils::DeviceMovableType::MEMMOV,
                      "should be memmov");

        MemmovClassExample obj;

        tg.task(SpWrite(obj),
            SpCuda([](SpDeviceDataView<MemmovClassExample> objv) {
            })
        );

        tg.waitAllTasks();
    }

    void SetTests() {
        Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
        Parent::AddTest(&SimpleGpuTest::TestVec, "Basic gpu test with vec");
        Parent::AddTest(&SimpleGpuTest::TestMemMove, "Basic gpu test with memmov");
    }
};

// You must do this
TestClass(SimpleGpuTest)
