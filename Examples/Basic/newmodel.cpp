///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

int main(){
    std::cout << "Example 1:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpRead(val), [&promise1](const int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("First-task");
        
        const int nbUncertainTasks = 2;

        for(int idx = 0 ; idx < nbUncertainTasks ; ++idx){
            runtime.potentialTask(SpMaybeWrite(val), [](int& /*valParam*/) -> bool {
                return false;
            }).setTaskName("Uncertain task -- " + std::to_string(idx));
        }

        runtime.task(SpWrite(val), [](int& valParam){
        }).setTaskName("Last-task");
        
        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex1-trace.svg");
        runtime.generateDot("/tmp/ex1-dag.dot");
    }
    return 0;
}


