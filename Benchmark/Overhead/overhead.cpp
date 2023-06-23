#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <algorithm>
#include <memory>
#include <limits>

#include "Utils/SpUtils.hpp"
#include "Legacy/SpRuntime.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"


auto estimate_overhead(const int NbLoops, const int NbThreads, const double SleepDuration){
    std::vector<int> data(NbThreads);

    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(NbThreads));
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
    tg.computeOn(ce);

    SpTimer totalTime;

    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        for(int idxThread = 0 ; idxThread < NbThreads ; ++idxThread){
            tg.task(SpPriority(1), SpWrite(data[idxThread]),
                    [SleepDuration](int&){
                        SpTimer timer;
                        timer.start();
                        do{
                            timer.stop();
                        } while(timer.getElapsed() < SleepDuration);
                    });
        }
    }

    tg.waitAllTasks();
    ce.stopIfNotAlreadyStopped();

    totalTime.stop();

    return totalTime.getElapsed() - SleepDuration*NbLoops;
}

int main(int argc, char** argv){
    const int MaxNbThreads = SpUtils::DefaultNumThreads();
    const int NbLoops = 10000;
    const double MaxTaskDuration = 10.;


    for(double sleepDuration = 0.00001 ; sleepDuration <= MaxTaskDuration ; sleepDuration *= 10){
        double maxOverheadPerTask = std::numeric_limits<double>::min();
        double avgOverheadPerTask = 0;
        int totalNbTasks = 0;

        for(int idxLoop = 1 ; idxLoop <= NbLoops ; ++idxLoop){
            for(int idxThread = 1 ; idxThread <= MaxNbThreads ; ++idxThread){
                const double duration = estimate_overhead(idxLoop, idxThread, sleepDuration);
                const double perTask = (duration/double(idxThread*idxLoop));
                maxOverheadPerTask = std::max(perTask, maxOverheadPerTask);
                avgOverheadPerTask += perTask;
                totalNbTasks += idxThread*idxLoop;
//                std::cout << sleepDuration << " " << idxThread << " "
//                          << idxLoop << " " << duration
//                          << " per task  " << perTask << std::endl;
            }
        }

        std::cout << "Max overhead = " << maxOverheadPerTask << " avg = " << avgOverheadPerTask/double(totalNbTasks) << std::endl;
    }


    return 0;
}
