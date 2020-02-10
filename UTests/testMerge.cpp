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

class TestMerge : public UTester< TestMerge > {
    using Parent = UTester< TestMerge >;
    
    template <SpSpeculativeModel Spm>
    void Test(){
        SpRuntime<Spm> runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });
        
        int a=0, b=0;

        std::promise<bool> promise1;
        std::promise<bool> promise2;

        runtime.potentialTask(SpMaybeWrite(a), [&promise1](int& param_a){
            promise1.get_future().get();
            return false;
        });
        
        runtime.potentialTask(SpMaybeWrite(b), [&promise2](int& param_b){
            promise2.get_future().get();
            return false;
        });
        
        runtime.potentialTask(SpMaybeWrite(a), SpMaybeWrite(b), [](int& param_a, int& param_b){
            return false;
        });
        
        promise1.set_value(true);
        promise2.set_value(true);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/testMerge" + std::to_string(static_cast<int>(Spm)) + ".dot");
    }
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }
    void Test2() { Test<SpSpeculativeModel::SP_MODEL_2>(); }
    void Test3() { Test<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&TestMerge::Test1, "Basic merge test for model 1");
        Parent::AddTest(&TestMerge::Test2, "Basic merge test for model 2");
        Parent::AddTest(&TestMerge::Test3, "Basic merge test for model 3");
    }
};

// You must do this
TestClass(TestMerge)


