///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <future>

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

class TestCommute : public UTester< TestCommute > {
    using Parent = UTester< TestCommute >;

    void TestBasic(){
        {
            SpRuntime runtime(10);
            std::promise<int> delay1;

            int dumbVal = 0;
            int commuteVal = 0;

            auto descr0 = runtime.task(SpWrite(dumbVal),
                         [&](int& dumbValParam){
                delay1.get_future().get();
                dumbValParam = 1;
            });
            UASSERTETRUE(descr0.isReady() == true);

            auto descr1 = runtime.task(SpRead(dumbVal), SpCommuteWrite(commuteVal),
                         [&](const int& dumbValParam, int& commuteValParam){
                UASSERTETRUE(dumbValParam == 1);
                UASSERTETRUE(commuteValParam == 1);
            });
            UASSERTETRUE(descr1.isReady() == false);

            auto descr2 = runtime.task(SpCommuteWrite(commuteVal),
                         [&](int& commuteValParam){
                UASSERTETRUE(commuteValParam == 0);
                commuteValParam = 1;
            });
            UASSERTETRUE(descr2.isReady() == true);

            delay1.set_value(0);

            runtime.waitAllTasks();
        }
        std::cout << "Next..." << std::endl;
        {
            SpRuntime runtime(2);

            std::atomic<int> initVal(0);

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpCommuteWrite(initVal),
                             [&](std::atomic<int>& initValParam){
                    UASSERTETRUE(initValParam == 0);
                    initValParam += 1;
                    usleep(1000);
                    UASSERTETRUE(initValParam == 1);
                    initValParam -= 1;
                });
            }

            runtime.waitAllTasks();
        }
        std::cout << "Next..." << std::endl;
        {
            SpRuntime runtime(10);

            std::atomic<int> initVal(0);

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpCommuteWrite(initVal),
                             [&](std::atomic<int>& initValParam){
                    UASSERTETRUE(initValParam == 0);
                    initValParam += 1;
                    usleep(1000);
                    UASSERTETRUE(initValParam == 1);
                    initValParam -= 1;
                });
            }

            runtime.waitAllTasks();
        }
    }

    void SetTests() {
        Parent::AddTest(&TestCommute::TestBasic, "Basic test for commute access");
    }
};

// You must do this
TestClass(TestCommute)


