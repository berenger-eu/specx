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

__global__ void inc_var(int* ptr){
    if(blockIdx.x == 0 && threadIdx.x == 0){
        (*ptr)++;
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
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)));
            #else
                        (*static_cast<int*>(std::get<0>(paramA)))++;
            #endif
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    })
        );

        tg.task(SpWrite(b),
                    SpCuda([](std::pair<void*, std::size_t> paramB) {
            #ifndef SPETABARU_EMUL_GPU
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramB)));
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
                        inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(static_cast<int*>(std::get<0>(paramA)));
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

    void SetTests() {
        Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
    }
};

// You must do this
TestClass(SimpleGpuTest)
