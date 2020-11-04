///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class SimpleTest : public UTester< SimpleTest > {
    using Parent = UTester< SimpleTest >;

    void TestBasic(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        const int initVal = 1;
        int writeVal = 0;

        runtime.task(SpRead(initVal),
                     [this, &initVal](const int& initValParam){
            UASSERTETRUE(&initValParam == &initVal);
        });

        runtime.task(SpRead(initVal), SpWrite(writeVal),
                     [](const int& initValParam, int& writeValParam){
            writeValParam += initValParam;
        });

        runtime.task(SpRead(writeVal),
                     [this](const int& writeValParam){
            UASSERTETRUE(writeValParam == initVal);
        });

        runtime.waitAllTasks();

        runtime.stopAllThreads();
    }

    void SetTests() {
        Parent::AddTest(&SimpleTest::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(SimpleTest)


