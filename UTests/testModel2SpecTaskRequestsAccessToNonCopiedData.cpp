///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class TestModel2SpecTaskRequestAccessToNonCopiedData : public UTester<TestModel2SpecTaskRequestAccessToNonCopiedData> {
    using Parent = UTester<TestModel2SpecTaskRequestAccessToNonCopiedData>;

    void Test(){
        int a=0, b=0;
        std::promise<bool> promise1;
        
        SpRuntime<SpSpeculativeModel::SP_MODEL_2> runtime;
        
        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        runtime.task(SpPotentialWrite(a), [&promise1]([[maybe_unused]] int &param_a) -> bool{
            promise1.get_future().get();
            return false;
        });
        
        runtime.task(SpRead(a), SpPotentialWrite(b), []([[maybe_unused]] const int &param_a, [[maybe_unused]] int &param_b) -> bool{
            return false;
        });
        
        promise1.set_value(true);
        
        runtime.waitAllTasks();
        runtime.stopAllThreads();
    }

    void SetTests() {
        Parent::AddTest(&TestModel2SpecTaskRequestAccessToNonCopiedData::Test, 
                        "Test behavior when speculative model 2 is used and a to be inserted speculative task requests access to non copied data");
    }
};

// You must do this
TestClass(TestModel2SpecTaskRequestAccessToNonCopiedData)
