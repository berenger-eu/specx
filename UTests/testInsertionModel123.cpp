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

class TestInsertionModel123 : public UTester< TestInsertionModel123 > {
    using Parent = UTester< TestInsertionModel123 >;
    
    template <SpSpeculativeModel Spm>
    void Test(){
        SpRuntime<Spm> runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });
        
        int a=0, b=0, c=0, d=0;

        std::promise<bool> promise1;

        runtime.task(SpWrite(a), [&promise1](int& param_a){
            param_a = 1;
            promise1.get_future().get();
        });

        runtime.potentialTask(SpRead(a), SpMaybeWrite(b), [](const int& a_param, int&) -> bool{
            return false;
        });

        runtime.potentialTask(SpRead(b), SpMaybeWrite(c), [](const int& param_b, int&) -> bool {
            bool res = false;
            if(param_b != 0) {
                res = true;
            }
            return res;
        });
        
        runtime.task(SpRead(c), SpWrite(d), [this](const int& param_c, int&param_d){
            if(param_c == 0) {
                param_d = 1;
            }
        });
        
        runtime.task(SpRead(d), [this](const int& param_d) {
            UASSERTEEQUAL(param_d, 1);
        });
        
        promise1.set_value(true);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/testIns" + std::to_string(static_cast<int>(Spm)) + ".dot");
    }
    
    void Test1() {
        return Test<SpSpeculativeModel::SP_MODEL_1>();
    }
    
    void Test2() {
        return Test<SpSpeculativeModel::SP_MODEL_2>();
    }
    
    void Test3() {
        return Test<SpSpeculativeModel::SP_MODEL_3>();
    }

    void SetTests() {
        Parent::AddTest(&TestInsertionModel123::Test1, "Basic insertion test for model 1");
        Parent::AddTest(&TestInsertionModel123::Test2, "Basic insertion test for model 2");
        Parent::AddTest(&TestInsertionModel123::Test3, "Basic insertion test for model 3");
    }
};

// You must do this
TestClass(TestInsertionModel123)


