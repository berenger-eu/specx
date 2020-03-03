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

class TestInsertion : public UTester< TestInsertion > {
    using Parent = UTester< TestInsertion >;
    
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

        runtime.task(SpRead(a), SpMaybeWrite(b), []([[maybe_unused]] const int& a_param, [[maybe_unused]] int&) -> bool{
            return false;
        });

        runtime.task(SpRead(b), SpMaybeWrite(c), [](const int& param_b, [[maybe_unused]] int&) -> bool {
            bool res = false;
            if(param_b != 0) {
                res = true;
            }
            return res;
        });
        
        runtime.task(SpRead(c), SpWrite(d), [](const int& param_c, int&param_d){
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
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }
    void Test2() { Test<SpSpeculativeModel::SP_MODEL_2>(); }
    void Test3() { Test<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&TestInsertion::Test1, "Basic insertion test for model 1");
        Parent::AddTest(&TestInsertion::Test2, "Basic insertion test for model 2");
        Parent::AddTest(&TestInsertion::Test3, "Basic insertion test for model 3");
    }
};

// You must do this
TestClass(TestInsertion)


