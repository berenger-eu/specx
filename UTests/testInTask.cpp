#include "Utils/SpUtils.hpp"
#include "Runtimes/SpRuntime.hpp"

#include "UTester.hpp"

class TestInTask : public UTester< TestInTask > {
    using Parent = UTester< TestInTask >;


    void TestBasic(){
        UASSERTETRUE(SpUtils::GetThreadId() == 0);
        UASSERTETRUE(SpUtils::IsInTask() == false);

        SpRuntime runtime(2);

        const int initVal = 1;

        runtime.task(SpRead(initVal),
                     [&](const int& /*initValParam*/){
            UASSERTETRUE(0 < SpUtils::GetThreadId());
            UASSERTETRUE(SpUtils::GetThreadId() <= 2);
            UASSERTETRUE(SpUtils::IsInTask() == true);
        });

        runtime.task(SpRead(initVal),
                     [&](const int& /*initValParam*/){
            UASSERTETRUE(0 < SpUtils::GetThreadId());
            UASSERTETRUE(SpUtils::GetThreadId() <= 2);
            UASSERTETRUE(SpUtils::IsInTask() == true);
        });

        runtime.task(SpRead(initVal),
                     [&](const int& /*initValParam*/){
            UASSERTETRUE(0 < SpUtils::GetThreadId());
            UASSERTETRUE(SpUtils::GetThreadId() <= 2);
            UASSERTETRUE(SpUtils::IsInTask() == true);
        });
    }

    void SetTests() {
        Parent::AddTest(&TestInTask::TestBasic, "Basic test for utils in task");
    }
};

// You must do this
TestClass(TestInTask)

