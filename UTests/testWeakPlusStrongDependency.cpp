///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class WeakPlusStrongDependency : public UTester< WeakPlusStrongDependency> {
    using Parent = UTester< WeakPlusStrongDependency >;
    
    template <SpSpeculativeModel Spm>
    void Test(){
        int a=0, b=0, c=0;
        
        std::promise<bool> promise1;
        
        SpRuntime<Spm> runtime;
        
        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        runtime.task(SpMaybeWrite(a), [&promise1]([[maybe_unused]] int &param_a) -> bool{
            promise1.get_future().get();
            return false;
        });
        
        runtime.task(SpRead(a), SpMaybeWrite(b), SpWrite(c), [](const int &param_a, int &param_b, int &param_c) -> bool{
            bool res = false;
            if(param_a != 0) {
               param_b = 1;
               res = true;
            }
            param_c = 1;
            return res;
        });
        
        runtime.task(SpRead(b), SpRead(c), [this](const int &param_b, [[maybe_unused]] const int &param_c){
            if(param_b == 0) {
                UASSERTEDIFF(param_c, 0);
            }
        });
        
        promise1.set_value(true);
        
        runtime.waitAllTasks();
        runtime.stopAllThreads();
    }
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }
    void Test2() { Test<SpSpeculativeModel::SP_MODEL_2>(); }
    void Test3() { Test<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&WeakPlusStrongDependency::Test1, "Test behavior when there are weak as well strong dependencies between speculative tasks (model 1)");
        Parent::AddTest(&WeakPlusStrongDependency::Test2, "Test behavior when there are weak as well strong dependencies between speculative tasks (model 2)");
        Parent::AddTest(&WeakPlusStrongDependency::Test3, "Test behavior when there are weak as well strong dependencies between speculative tasks (model 3)");
    }
};

// You must do this
TestClass(WeakPlusStrongDependency)
