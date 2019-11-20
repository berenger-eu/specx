///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class TestSelectActivationLogicModel1 : public UTester< TestSelectActivationLogicModel1> {
    using Parent = UTester< TestSelectActivationLogicModel1 >;

    void Test(){
        int a=0, b=0;
        SpRuntime<SpSpeculativeModel::SP_MODEL_1> runtime(SpUtils::DefaultNumThreads());

        runtime.potentialTask(SpMaybeWrite(a), [](int &param_a) -> bool{
            (void) param_a;
            return false;
        });
        
        runtime.potentialTask(SpRead(a), SpMaybeWrite(b), [](const int &param_a, int &param_b) -> bool{
            bool res = false;
            if(param_a == 0) {
               param_b = 1;
               res = true;
            }
            return res;
        });
        
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        
        UASSERTEEQUAL(b, 1);
    }

    void SetTests() {
        Parent::AddTest(&TestSelectActivationLogicModel1::Test, 
                        "Test behavior of select activation logic in model 1");
    }
};

// You must do this
TestClass(TestSelectActivationLogicModel1)
