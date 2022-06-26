///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"

class SimpleTestNoSpec : public UTester< SimpleTestNoSpec > {
    using Parent = UTester< SimpleTestNoSpec >;

    void TestBasic(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        
        tg.computeOn(ce);

        const int initVal = 1;
        int writeVal = 0;

        tg.task(SpRead(initVal),
                     [this, &initVal](const int& initValParam){
            UASSERTETRUE(&initValParam == &initVal);
        });

        tg.task(SpRead(initVal), SpWrite(writeVal),
                     [](const int& initValParam, int& writeValParam){
            writeValParam += initValParam;
        });

        tg.task(SpRead(writeVal),
                     [this](const int& writeValParam){
            UASSERTETRUE(writeValParam == initVal);
        });

        tg.waitAllTasks();
    }

    void SetTests() {
        Parent::AddTest(&SimpleTestNoSpec::TestBasic, "Basic test for non speculative task graphs");
    }
};

// You must do this
TestClass(SimpleTestNoSpec)


