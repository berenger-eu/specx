///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorker.hpp"
#include "TaskGraph/SpTaskGraph.hpp"

class SimpleGpuTest : public UTester< SimpleGpuTest > {
    using Parent = UTester< SimpleGpuTest >;

    void Test(){
        SpComputeEngine ce(SpWorker::createHeterogeneousTeamOfWorkers(1,1));
        SpTaskGraph tg;
        int a = 0;
        
        tg.computeOn(ce);
        
        tg.task(
        SpWrite(a),
        SpCpu(
        [](int& paramA) {
            paramA++;
        })
        );
        
        tg.task(
        SpWrite(a),
        SpCpu(
        [](int& paramA) {
            paramA++;
        }),
        SpGpu(
        [](std::pair<void*, std::size_t> paramA) {
            (*static_cast<int*>(std::get<0>(paramA)))++;
        })
        );
        
        tg.task(
        SpWrite(a),
        SpGpu(
        [](std::pair<void*, std::size_t> paramA) {
            (*static_cast<int*>(std::get<0>(paramA)))++;
        })
        );
        
        tg.waitAllTasks();
        UASSERTETRUE(a == 3);
    }

    void SetTests() {
        Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
    }
};

// You must do this
TestClass(SimpleGpuTest)
