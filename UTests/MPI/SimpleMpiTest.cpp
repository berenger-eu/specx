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

class SimpleMpiTest : public UTester< SimpleMpiTest > {
    using Parent = UTester< SimpleMpiTest >;

    void Test(){
        SpMpiBackgroundWorker::GetWorker().init();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        int a = 1;
        int b = 0;

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const int& paramA, int& paramB) {
                            paramB = paramA + paramB;
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const int& paramA, int& paramB) {
                            paramB = paramA + paramB;
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(b == 2);
    }


    void SetTests() {
        Parent::AddTest(&SimpleMpiTest::Test, "Basic MPI test");
    }
};

// You must do this
TestClass(SimpleMpiTest)
