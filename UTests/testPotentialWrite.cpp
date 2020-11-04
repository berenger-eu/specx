///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"
#include "utestUtils.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpArrayAccessor.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class TestPotentialWrite : public UTester< TestPotentialWrite > {
    using Parent = UTester< TestPotentialWrite >;

    template <SpSpeculativeModel Spm>
    void TestBasic(){
        SpRuntime<Spm> runtime;


        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        int values0 = -1;
        int values1 = -1;
        int values2 = -1;

        std::promise<int> promise1;

        runtime.task(SpWrite(values0), SpWrite(values1), SpWrite(values2),
                [&promise1](int& values0param, int& values1param, int& values2param){
            promise1.get_future().get();
            values0param = 1;
            values1param = 2;
            values2param = 3;
        });

        runtime.task(SpPotentialWrite(values0), SpPotentialWrite(values1), SpRead(values2),
                [this](int& values0param, int& values1param, const int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return false;
        });

        runtime.task(SpPotentialWrite(values0), SpRead(values1), SpPotentialWrite(values2),
                [this](int& values0param, const int& values1param, int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return false;
        });

        runtime.task(SpWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0param, int& values1param, int& values2param){
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
        });

        runtime.task(SpPotentialWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0param, int& values1param, int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return false;
        });

        runtime.task(SpWrite(values0), SpPotentialWrite(values1), SpPotentialWrite(values2),
                [this](int& values0param, int& values1param, int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return false;
        });

        runtime.task(SpPotentialWrite(values0), SpWrite(values1), SpWrite(values2),
                [this](int& values0param, int& values1param, int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return true;
        });

        runtime.task(SpWrite(values0), SpPotentialWrite(values1), SpPotentialWrite(values2),
                [this](int& values0param, int& values1param, int& values2param) -> bool {
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
            return true;
        });

        runtime.task(SpRead(values0), SpRead(values1), SpRead(values2),
                [this](const int& values0param, const int& values1param, const int& values2param){
            UASSERTEEQUAL(values0param, 1);
            UASSERTEEQUAL(values1param, 2);
            UASSERTEEQUAL(values2param, 3);
        });

        promise1.set_value(0);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/test.dot");
        runtime.generateTrace("/tmp/test.svg");
    }
    
    void TestBasic1() { TestBasic<SpSpeculativeModel::SP_MODEL_1>(); }
    void TestBasic2() { TestBasic<SpSpeculativeModel::SP_MODEL_2>(); }
    void TestBasic3() { TestBasic<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&TestPotentialWrite::TestBasic1, "Basic test for vec type");
        Parent::AddTest(&TestPotentialWrite::TestBasic2, "Basic test for vec type");
        Parent::AddTest(&TestPotentialWrite::TestBasic3, "Basic test for vec type");
    }
};

// You must do this
TestClass(TestPotentialWrite)


