///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <future>

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class TestWindow : public UTester< TestWindow > {
    using Parent = UTester< TestWindow >;

    void TestBasic(){
        {
            const int NbTasks = 10;
            std::promise<int> promises[NbTasks];

            const int NbThreads = 10;
            SpRuntime runtime(NbThreads);

            int readVal = 0;

            for(int idxTask = 0 ; idxTask < NbTasks ; ++idxTask){
                runtime.task(SpRead(readVal),
                             [&,idxTask](const int& /*readVal*/){
                    promises[idxTask].get_future().get();
                    usleep(1000);
                });
            }

            for(int idxTask = NbTasks ; idxTask < 2*NbTasks ; ++idxTask){
                runtime.waitRemain(idxTask);
            }

            for(int idxTaskPromise = 0 ; idxTaskPromise < NbTasks ; ++idxTaskPromise){
                promises[idxTaskPromise].set_value(0);
                for(int idxTask = NbTasks-idxTaskPromise-1 ; idxTask < 2*NbTasks ; ++idxTask){
                    runtime.waitRemain(idxTask);
                }
            }

            runtime.waitAllTasks();
        }
        {
            const int NbThreads = 10;
            SpRuntime runtime(NbThreads);

            int readVal = 0;
            std::promise<int> promise0;
            std::promise<int> promise1;

            runtime.task(SpRead(readVal),
                         [&](const int& /*readVal*/){
                promise0.get_future().get();
            });

            runtime.task(SpRead(readVal),
                         [&](const int& /*readVal*/){
                promise1.get_future().get();
            });

            runtime.waitRemain(3);

            promise0.set_value(0);
            runtime.waitRemain(2);

            promise1.set_value(0);

            runtime.waitAllTasks();
        }
    }

    void SetTests() {
        Parent::AddTest(&TestWindow::TestBasic, "Basic test for commute access");
    }
};

// You must do this
TestClass(TestWindow)


