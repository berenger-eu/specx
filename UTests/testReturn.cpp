///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class ReturnTest : public UTester< ReturnTest > {
    using Parent = UTester< ReturnTest >;

    void TestBasic(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        const int initVal = 1;
        int writeVal = 0;

        {
            auto returnValue = runtime.task(SpRead(initVal),
                         [this, &initVal](const int& initValParam){
                UASSERTETRUE(&initValParam == &initVal);
            });
            returnValue.getValue();
        }

        {
            auto returnValue = runtime.task(SpRead(initVal), SpWrite(writeVal),
                         [](const int& initValParam, int& writeValParam) -> bool {
                writeValParam += initValParam;
                return true;
            });
            returnValue.wait();
            UASSERTETRUE(returnValue.getValue() == true);
        }

        {
            auto returnValue = runtime.task(SpRead(writeVal),
                         [this](const int& writeValParam) -> int {
                UASSERTETRUE(writeValParam == initVal);
                return 99;
            });
            UASSERTETRUE(returnValue.getValue() == 99);
        }

        runtime.waitAllTasks();

        runtime.stopAllThreads();
    }

    void SetTests() {
        Parent::AddTest(&ReturnTest::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(ReturnTest)


