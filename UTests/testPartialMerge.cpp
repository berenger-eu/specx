///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
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

class TestPartialMerge : public UTester< TestPartialMerge > {
    using Parent = UTester< TestPartialMerge >;
    
    template <SpSpeculativeModel Spm>
    void Test(){
        SpRuntime<Spm> runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });
        
        int a=0, b=0, c=0;

        std::promise<bool> promise1;

        runtime.task(SpPotentialWrite(a), SpPotentialWrite(b), [&promise1]([[maybe_unused]] int& param_a, [[maybe_unused]] int& param_b){
            promise1.get_future().get();
            return false;
        });
        
        runtime.task(SpPotentialWrite(c), []([[maybe_unused]] int& param_c){
            return false;
        });
        
        runtime.task(SpPotentialWrite(a), SpPotentialWrite(c), []([[maybe_unused]] int& param_a, [[maybe_unused]] int& param_c){
            return false;
        });
        
        promise1.set_value(true);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/testPartialMerge" + std::to_string(static_cast<int>(Spm)) + ".dot", true);
    }
    
    void Test1() { Test<SpSpeculativeModel::SP_MODEL_1>(); }
    void Test2() { Test<SpSpeculativeModel::SP_MODEL_2>(); }
    void Test3() { Test<SpSpeculativeModel::SP_MODEL_3>(); }

    void SetTests() {
        Parent::AddTest(&TestPartialMerge::Test1, "Partial merge test for model 1");
        Parent::AddTest(&TestPartialMerge::Test2, "Partial merge test for model 2");
        Parent::AddTest(&TestPartialMerge::Test3, "Partial merge test for model 3");
    }
};

// You must do this
TestClass(TestPartialMerge)


