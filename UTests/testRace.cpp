///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"
#include "utestUtils.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class TestRace : public UTester< TestRace > {
    using Parent = UTester< TestRace >;

    void TestBasic(){
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
                runtime.task(SpWrite(val[idx]),
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
        Parent::AddTest(&TestRace::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(TestRace)


