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

class TestParallelWrite : public UTester< TestParallelWrite > {
    using Parent = UTester< TestParallelWrite >;

    void TestBasic(){
        {
            SpRuntime runtime(2);

            std::atomic<int> initVal(0);

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpParallelWrite(initVal),
                             [&](std::atomic<int>& initValParam){
                    initValParam += 1;
                    while(initValParam != runtime.getNbThreads()){
                        usleep(100);
                    }
                });
            }

            runtime.waitAllTasks();
        }
        {
            SpRuntime runtime(10);

            std::atomic<int> initVal(0);

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpParallelWrite(initVal),
                             [&](std::atomic<int>& initValParam){
                    initValParam += 1;
                    while(initValParam != runtime.getNbThreads()){
                        usleep(100);
                    }
                });
            }

            runtime.waitAllTasks();
        }
        {
            SpRuntime runtime(10);
            std::promise<long int> promises[10];

            int dumbVal = 0;

            for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
                runtime.task(SpParallelWrite(dumbVal),
                             [&,idxThread](int& /*dumbValParam*/){
                    promises[idxThread].set_value(idxThread);
                    const long int res = promises[(idxThread+1)%10].get_future().get();
                    UASSERTETRUE(res == (idxThread+1)%10);
                });
            }

            runtime.waitAllTasks();
        }
    }

    void SetTests() {
        Parent::AddTest(&TestParallelWrite::TestBasic, "Basic test for parrallel write access");
    }
};

// You must do this
TestClass(TestParallelWrite)


