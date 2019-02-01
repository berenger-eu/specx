///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"
#include "utestUtils.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpArrayAccessor.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class TestMaybeWrite : public UTester< TestMaybeWrite > {
    using Parent = UTester< TestMaybeWrite >;

    void TestBasic(){
        SpRuntime runtime;


        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        int values0 = -1;
        int values1 = -1;
        int values2 = -1;

        std::promise<int> promise1;

        runtime.task(SpWrite(values0), SpWrite(values1), SpWrite(values2),
                [&promise1](int& values0, int& values1, int& values2){
            promise1.get_future().get();
            values0 = 1;
            values1 = 2;
            values2 = 3;
        });

        runtime.potentialTask(SpMaybeWrite(values0), SpMaybeWrite(values1), SpRead(values2),
                [this](int& values0, int& values1, const int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return false;
        });

        runtime.potentialTask(SpMaybeWrite(values0), SpRead(values1), SpMaybeWrite(values2),
                [this](int& values0, const int& values1, int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return false;
        });

        runtime.task(SpWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0, int& values1, int& values2){
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
        });

        runtime.potentialTask(SpMaybeWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0, int& values1, int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return false;
        });

        runtime.potentialTask(SpWrite(values0), SpMaybeWrite(values1), SpMaybeWrite(values2),
                [this](int& values0, int& values1, int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return false;
        });

        runtime.potentialTask(SpMaybeWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0, int& values1, int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return true;
        });

        runtime.potentialTask(SpWrite(values0), SpMaybeWrite(values1), SpMaybeWrite(values2),
                [this](int& values0, int& values1, int& values2) -> bool {
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
            return true;
        });

        runtime.task(SpRead(values0), SpRead(values1), SpRead(values2),
                [this](const int& values0, const int& values1, const int& values2){
            UASSERTEEQUAL(values0, 1);
            UASSERTEEQUAL(values1, 2);
            UASSERTEEQUAL(values2, 3);
        });

        promise1.set_value(0);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/test.dot");
        runtime.generateTrace("/tmp/test.svg");
    }

    void SetTests() {
        Parent::AddTest(&TestMaybeWrite::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(TestMaybeWrite)


