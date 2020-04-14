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

class TestCallableWrappers : public UTester< TestCallableWrappers > {
    using Parent = UTester< TestCallableWrappers >;
    
    template <SpSpeculativeModel Spm>
    void Test(){
        SpRuntime<Spm> runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });
        
        int a=0;

        runtime.task(SpWrite(a), [](int& param_a){
            param_a++;
        });
        
        runtime.task(SpWrite(a), SpCpu([](int& param_a){
            param_a++;
        }));
        
        runtime.task(SpWrite(a),
        [](int& param_a){
            param_a++;
        },
        SpGpu([](int& param_a){
            param_a++;
        }));
        
        runtime.task(SpWrite(a),
        SpCpu([](int& param_a){
            param_a++;
        }),
        SpGpu([](int& param_a){
            param_a++;
        }));

        runtime.task(SpWrite(a),
        SpGpu([](int& param_a){
            param_a++;
        }),
        [](int& param_a){
            param_a++;
        });
        
        runtime.task(SpWrite(a),
        SpGpu([](int& param_a){
            param_a++;
        }),
        SpCpu([](int& param_a){
            param_a++;
        }));
                
        runtime.waitAllTasks();
        runtime.stopAllThreads();

        UASSERTETRUE(a == 6);

    }
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }

    void SetTests() {
        Parent::AddTest(&TestCallableWrappers::Test1, "Test callable wrappers.");
    }
};

// You must do this
TestClass(TestCallableWrappers)


