///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"

int main(){
    // The workers
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());

    // The normal task graph
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
    // The detach task graph
    SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tgDetach;

    // Assign the workers to the graphs
    tg.computeOn(ce);
    tgDetach.computeOn(ce);

    int fakeData = 1;

    // A normal task, with classical dependencies
    tg.task(SpWrite(fakeData), [&tgDetach](int& depFakeData){
        std::cout << "Start main task..." << std::endl;
        // Variant 1: we insert a task in tgDetach using dependencies
        // This method should not be used if there is a risk that another
        // detached task could be submited before this one ends!
        // Moreover, one must be sure that depFakeData remains alive
        // until this task is over!
        tgDetach.task(SpRead(depFakeData), []([[maybe_unused]] const int& depFakeDataDetach){
            // Do whatever you want with depFakeDataDetach
            std::cout << "Start variant 1 task..." << std::endl;
            sleep(1);
            std::cout << "End variant 1 task." << std::endl;
        });

        // Variant 2: we insert a task in tgDetach WITHOUT dependencies
        // However, here pass a reference of depFakeData, consequently
        // as with Variant 1, we must be sure that depFakeData is alive
        // until this task is over
        tgDetach.task([&depFakeData](){
            // Do whatever you want with depFakeDataDetach
            std::cout << "Start variant 2 task..." << std::endl;
            sleep(1);
            std::cout << "End variant 2 task." << std::endl;
        });

        // Variant 3: we insert a task in tgDetach WITHOUT dependencies
        // and we copy depFakeData, so it is safe if we detach again
        // a task on depFakeData or if depFakeData is destroyed before
        // this task ends. But it is at the cost of a copy (which could be
        // expensive).
        tgDetach.task([depFakeData](){
            // Do whatever you want with depFakeDataDetach
            std::cout << "Start variant 3 task..." << std::endl;
            sleep(1);
            std::cout << "End variant 3 task." << std::endl;
        });
        std::cout << "End main task." << std::endl;
    });

    // Whenever you need, sync with tg
    tg.waitAllTasks();

    // At the end of the application
    tgDetach.waitAllTasks();
    ce.stopIfNotAlreadyStopped();

    return 0;
}
