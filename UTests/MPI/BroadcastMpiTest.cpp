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
#include "MPI/SpMpiUtils.hpp"

class BroadcastMpiTest : public UTester< BroadcastMpiTest > {
    using Parent = UTester< BroadcastMpiTest >;

    void Test(){
        SpMpiBackgroundWorker::GetWorker().init();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        int a = 1;
        int b = 1;

        tg.computeOn(ce);

        // This test works with only 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const int& paramA, int& paramB) {
                            paramB = paramA + paramB;
                        })
            );

            tg.mpiBroadcastSend(b, 0);
            tg.mpiBroadcastRecv(a, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const int& paramA, int& paramB) {
                            paramB = paramA + paramB;
                        })
            );

            tg.mpiBroadcastRecv(a, 0);
            tg.mpiBroadcastSend(b, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(a == 2);
        UASSERTETRUE(b == 2);
    }


    void SetTests() {
        Parent::AddTest(&BroadcastMpiTest::Test, "Basic Broadcast MPI test");
    }
};

// You must do this
TestClass(BroadcastMpiTest)
