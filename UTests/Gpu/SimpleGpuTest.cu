///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
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

class SimpleGpuTest : public UTester< SimpleGpuTest > {
    using Parent = UTester< SimpleGpuTest >;

    void Test(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        int a = 0;
        int b = 0;

        tg.computeOn(ce);

        tg.task(SpWrite(a),
                    SpCuda([](std::pair<void*, std::size_t> paramA) {
            #ifndef SPETABARU_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)), 1);
            #else
                        (*static_cast<int*>(std::get<0>(paramA)))++;
            #endif
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(b),
                    SpCuda([](std::pair<void*, std::size_t> paramB) {
            #ifndef SPETABARU_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramB)), 1);
            #else
                        (*static_cast<int*>(std::get<0>(paramB)))++;
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
                        [](std::pair<void*, std::size_t> paramA) {
            #ifndef SPETABARU_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)), 1);
            #else
                        (*static_cast<int*>(std::get<0>(paramA)))++;
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

        tg.computeOn(ce);

        tg.task(SpWrite(a),
                    SpCuda([](std::pair<void*, std::size_t> paramA) {
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)),
                                                                           std::get<1>(paramA)/sizeof(int));

                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(b),
                    SpCuda([](std::pair<void*, std::size_t> paramB) {
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramB)),
                                                                            std::get<1>(paramB)/sizeof(int));
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
                    SpCuda(
                        [](std::pair<void*, std::size_t> paramA) {
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)),
                                                                           std::get<1>(paramA)/sizeof(int));
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

    void SetTests() {
        Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
        Parent::AddTest(&SimpleGpuTest::TestVec, "Basic gpu test with vec");
    }
};

// You must do this
TestClass(SimpleGpuTest)
