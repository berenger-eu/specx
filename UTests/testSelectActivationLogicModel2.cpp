///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class TestSelectActivationLogicModel2 : public UTester< TestSelectActivationLogicModel2> {
    using Parent = UTester< TestSelectActivationLogicModel2 >;

    void Test(){
        int a=0, b=0;
        SpRuntime<SpSpeculativeModel::SP_MODEL_2> runtime(SpUtils::DefaultNumThreads());

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
        
        runtime.generateDot("/tmp/test.dot");
        
        UASSERTEEQUAL(b, 1);
    }

    void SetTests() {
        Parent::AddTest(&TestSelectActivationLogicModel2::Test, 
                        "Test behavior of select activation in model 2");
    }
};

// You must do this
TestClass(TestSelectActivationLogicModel2)
