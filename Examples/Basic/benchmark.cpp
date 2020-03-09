///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"
#include "Utils/SpTimer.hpp"
#include "Utils/small_vector.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    {
        const int NbTasksToSubmit = 1000;
        int initVal = 1;

        SpTimer timer;

        for(int idx = 0 ; idx < NbTasksToSubmit ; ++idx){
            runtime.task(SpRead(initVal),
                         [](const int&){
                usleep(1000);
            });
        }

        timer.stop();

        runtime.waitAllTasks();

        std::cout << "Average time to insert a task with no pressure = " << timer.getElapsed()/double(NbTasksToSubmit) << "s" << std::endl;
    }
    {
        const int NbTasksToSubmit = 1000;
        int initVal = 1;

        SpTimer timer;

        for(int idx = 0 ; idx < NbTasksToSubmit ; ++idx){
            runtime.task(SpRead(initVal),
                         [](const int&){
            });
        }

        timer.stop();

        runtime.waitAllTasks();

        std::cout << "Average time to insert a task with pressure = " << timer.getElapsed()/double(NbTasksToSubmit) << "s" << std::endl;
    }

    {
        static const int NbLoops = 100;
        const int NbTasksToSubmit = runtime.getNbThreads();
        int initVal = 1;

        small_vector<SpAbstractTaskWithReturn<double>::SpTaskViewer> elapsed;
        elapsed.reserve(NbTasksToSubmit*NbLoops);

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            for(int idx = 0 ; idx < NbTasksToSubmit ; ++idx){
                SpTimer timerTask;
                elapsed.emplace_back( runtime.task(SpRead(initVal),
                             [timerTask](const int&) mutable -> double {
                    timerTask.stop();
                    usleep(1000);
                    return timerTask.getElapsed();
                }) );
            }
            runtime.waitAllTasks();
        }


        double averageToExecute = 0;
        for(const SpAbstractTaskWithReturn<double>::SpTaskViewer& viewer : elapsed){
            averageToExecute += viewer.getValue()/double(NbTasksToSubmit*NbLoops);
        }

        std::cout << "Average time for a task to be executed without pressure = " << averageToExecute << "s" << std::endl;
    }
    {
        static const int NbLoops = 100;
        const int NbTasksToSubmit = runtime.getNbThreads();
        int initVal = 1;

        small_vector<SpAbstractTaskWithReturn<double>::SpTaskViewer> elapsed;
        elapsed.reserve(NbTasksToSubmit*NbLoops);

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            for(int idx = 0 ; idx < NbTasksToSubmit ; ++idx){
                SpTimer timerTask;
                elapsed.emplace_back( runtime.task(SpRead(initVal),
                             [timerTask](const int&) mutable -> double {
                    timerTask.stop();
                    return timerTask.getElapsed();
                }) );
            }
            runtime.waitAllTasks();
        }


        double averageToExecute = 0;
        for(const SpAbstractTaskWithReturn<double>::SpTaskViewer& viewer : elapsed){
            averageToExecute += viewer.getValue()/double(NbTasksToSubmit*NbLoops);
        }

        std::cout << "Average time for a task to be executed with pressure = " << averageToExecute << "s" << std::endl;
    }

    runtime.stopAllThreads();

    return 0;
}
