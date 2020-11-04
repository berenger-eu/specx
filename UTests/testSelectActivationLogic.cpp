///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class TestSelectActivationLogic : public UTester< TestSelectActivationLogic> {
    using Parent = UTester< TestSelectActivationLogic >;

    template <SpSpeculativeModel Spm>
    void Test(){
        int a=0, b=0;
        std::promise<bool> promise1;
        
        SpRuntime<Spm> runtime;
        
        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        runtime.task(SpPotentialWrite(a), [&promise1]([[maybe_unused]] int &param_a) -> bool{
            promise1.get_future().get();
            return false;
        });
        
        runtime.task(SpRead(a), SpPotentialWrite(b), [](const int &param_a, int &param_b) -> bool{
            bool res = false;
            if(param_a == 0) {
               param_b = 1;
               res = true;
            }
            return res;
        });
        
        promise1.set_value(true);
        
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        
        UASSERTEEQUAL(b, 1);
    }
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }
    void Test2() { Test<SpSpeculativeModel::SP_MODEL_2>(); }
    void Test3() { Test<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&TestSelectActivationLogic::Test1, "Test behavior of select activation logic in model 1");
        Parent::AddTest(&TestSelectActivationLogic::Test2, "Test behavior of select activation logic in model 2");
        Parent::AddTest(&TestSelectActivationLogic::Test3, "Test behavior of select activation logic in model 3");
    }
};

// You must do this
TestClass(TestSelectActivationLogic)
