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

class TestPotiential : public UTester< TestPotiential > {
    using Parent = UTester< TestPotiential >;

    void TestBasic(){
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpRead(val), [](const int& /*valParam*/){
        });
        // val is 0

        runtime.task(SpRead(val), [&promise1](const int& /*valParam*/){
            promise1.get_future().get();
        });
        // val is 0

        runtime.potentialTask(SpMaybeWrite(val), [](int& /*valParam*/) -> bool {
            std::cout << "Maybe task will return false" << std::endl;
            std::cout.flush();
            return false;
        });
        // val is 0

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), [&val,&counterFirstSpec](int& valParam){
            std::cout << "Speculative task, valParam is " << valParam << " at " << &valParam << std::endl;
            std::cout << "Speculative task, val is " << val << " at " << &val << std::endl;
            valParam += 1;
            std::cout << "Speculative task, valParam is " << valParam << " at " << &valParam << std::endl;
            std::cout << "Speculative task, val is " << val << " at " << &val << std::endl;
            std::cout.flush();
            counterFirstSpec += 1;
        });
        // val is 1


        runtime.potentialTask(SpMaybeWrite(val), [](int& valParam) -> bool {
            // valParam should be 1
            std::cout << "Maybe task 2, valParam is " << valParam << " at " << &valParam << std::endl;
            std::cout.flush();
            valParam += 2;
            std::cout << "Maybe task 2, return true with valParam is " << valParam << " at " << &valParam << std::endl;
            return true;
        });
        // val is 3

        std::atomic_int counterSecondSpec(0);

        runtime.task(SpWrite(val), [&val,&counterSecondSpec](int& valParam){
            std::cout << "Speculative last write, valParam is " << valParam << " at " << &valParam << std::endl;
            std::cout << "Speculative last write, val is " << val << " at " << &val << std::endl;
            valParam *= 2;
            std::cout << "Speculative last write, valParam is " << valParam << " at " << &valParam << std::endl;
            std::cout << "Speculative last write, val is " << val << " at " << &val << std::endl;
            std::cout.flush();
            counterSecondSpec += 1;
        });
        // val is 6

        promise1.set_value(0);

        runtime.waitAllTasks();
        runtime.stopAllThreads();

        runtime.generateDot("/tmp/test.dot");
        runtime.generateTrace("/tmp/test.svg");

        UASSERTEEQUAL(counterFirstSpec.load(), 1);
        UASSERTEEQUAL(counterSecondSpec.load(), 2);
        UASSERTEEQUAL(val, 6);
    }


    void TestBasicLoop(){
        std::array<unsigned int,2> SleepTimes{0, 500000};
        for(auto SleepTime : SleepTimes){
            SpRuntime runtime;

            runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
                return true;
            });

            const int arraySize = 6;
            int val[arraySize] = {0};

            UTestRaceChecker counterAccess;

            runtime.task(SpReadArray(val,SpArrayView(arraySize)), [](SpArrayAccessor<const int>& /*valParam*/){
            });
            // val is 0

            for(int idx = 0 ; idx < arraySize ; ++idx){
                runtime.potentialTask(SpMaybeWrite(val[idx]),
                                      SpReadArray(val,SpArrayView(arraySize).removeItem(idx)),
                                      [SleepTime,idx,&counterAccess]
                                      (int& valParam, const SpArrayAccessor<const int>& valArray) -> bool {
                    {
                        counterAccess.lock();
                        counterAccess.addWrite(&valParam);
                        for(int idxTest = 0 ; idxTest < valArray.getSize() ; ++idxTest){
                            counterAccess.addRead(&valArray.getAt(idxTest));
                        }
                        counterAccess.unlock();
                    }

                    if(idx == 3){
                        valParam += 1;
                    }
                    if(idx == 5){
                        valParam += 10;
                    }
                    usleep(SleepTime);

                    {
                        counterAccess.lock();
                        counterAccess.releaseWrite(&valParam);
                        for(int idxTest = 0 ; idxTest < valArray.getSize() ; ++idxTest){
                            counterAccess.releaseRead(&valArray.getAt(idxTest));
                        }
                        counterAccess.unlock();
                    }

                    return idx == 3 || idx == 5;
                });
            }

            runtime.waitAllTasks();
            runtime.stopAllThreads();

            UASSERTEEQUAL(val[3], 1);
            UASSERTEEQUAL(val[5], 10);
        }
    }

    void SetTests() {
        Parent::AddTest(&TestPotiential::TestBasic, "Basic test for vec type");
        // TODO Parent::AddTest(&TestPotiential::TestBasicLoop, "Basic test for vec type");
    }
};

// You must do this
TestClass(TestPotiential)


