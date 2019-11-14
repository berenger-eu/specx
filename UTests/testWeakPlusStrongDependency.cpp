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

    void Test(){
        int a=0, b=0, c=0;
        SpRuntime<SpSpeculativeModel::SP_MODEL_1> runtime(SpUtils::DefaultNumThreads());

        runtime.potentialTask(SpMaybeWrite(a), [](int &param_a) -> bool{
            (void) param_a;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            return false;
        });
        
        runtime.potentialTask(SpRead(a), SpMaybeWrite(b), SpWrite(c), [](const int &param_a, int &param_b, int &param_c) -> bool{
            bool res = false;
            if(param_a != 0) {
               param_b = 1;
               res = true;
            }
            param_c = 1;
            return res;
        });
        
        runtime.task(SpRead(b), SpRead(c), [this](const int &param_b, const int &param_c){
            if(param_b == 0) {
                UASSERTEDIFF(param_c, 0);
            }
        });
        
        runtime.waitAllTasks();
        runtime.stopAllThreads();
        runtime.generateDot("/tmp/graph.dot");
    }

    void SetTests() {
        Parent::AddTest(&WeakPlusStrongDependency::Test, 
                        "Test behavior when there are weak as well strong dependencies between speculative tasks");
    }
};

// You must do this
TestClass(WeakPlusStrongDependency)
