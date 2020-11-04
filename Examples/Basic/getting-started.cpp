///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <random>
#include <cassert>

#include "Data/SpDataAccessMode.hpp"
#include "Task/SpPriority.hpp"
#include "Task/SpProbability.hpp"
#include "Legacy/SpRuntime.hpp"

[[maybe_unused]] const size_t seedSpeculationSuccess = 42;
[[maybe_unused]] const size_t seedSpeculationFailure = 0;
const size_t seed = seedSpeculationSuccess;

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]){
    // First we instantiate a runtime object and we specify that the
    // runtime should use speculation model 2.
    SpRuntime<SpSpeculativeModel::SP_MODEL_2> runtime;
    
    // Next we set a predicate that will be called by the runtime each
    // time a speculative task becomes ready to run. It is used to
    // decide if the speculative task should be allowed to run.
    runtime.setSpeculationTest(
    []([[maybe_unused]] const int nbReadyTasks,
    [[maybe_unused]] const SpProbability& meanProbability) -> bool {
        return true; // Here we always return true, this basically means
                     // that we always allow speculative tasks to run
                     // regardless of runtime conditions.
    });

    int a = 41, b = 0, c = 0;

    // We create our first task. We are specifying that the task will be
    // reading from a. The task will call the lambda given as a last
    // argument to the call. The return value of the task is the return
    // value of the lambda.  
    auto task1 = runtime.task(SpRead(a), [](const int& inA) -> int {
        return inA + 1;
    });
    
    // Here we set a custom name for the task.
    task1.setTaskName("First-task");
    
    // Here we wait until task1 is finished and we retrieve its return
    // value. 
    b = task1.getValue();
    
    // Next we create a potential task, i.e. a task which might write to
    // some data.
    // In this case the task may write to "a" with a probability of 0.5.
    // Subsequent tasks will be allowed to speculate over this task.
    // The task returns a boolean to inform the runtime of whether or 
    // not it has written to its maybe-write data dependency a.
    std::mt19937_64 mtEngine(seed);
    std::uniform_real_distribution<double> dis01(0,1);
    
    runtime.task(SpPriority(0), SpProbability(0.5), SpRead(b),
    SpPotentialWrite(a),
    [dis01, mtEngine] (const int &inB, int &inA) mutable -> bool{
        double val = dis01(mtEngine);
        
        if(inB == 42  && val < 0.5) {
            inA = 43;
            return true;
        }
        
        return false;
        
    }).setTaskName("Second-task");
    
    // We create a final normal task that reads from a and writes to c.
    // The task reads from a so there should be a strict write -> read
    // dependency between the second and the final task but since the
    // second task may not always write to a, the runtime will try to
    // execute a speculative version of the final task in parallel
    // with the second task in case the second task doesn't write to a.
    runtime.task(SpRead(a), SpWrite(c), [] (const int &inA, int &inC) {
        if(inA == 41) {
            inC = 1;
        } else {
            inC = 2;
        }
    }).setTaskName("Final-task");

    // We wait for all tasks to finish
    runtime.waitAllTasks();
    
    // We make all runtime threads exit
    runtime.stopAllThreads();
    
    assert((a == 41 || a == 43) && b == 42 && (c == 1 || c == 2)
            && "Try again!");
    
    // We generate the task graph corresponding to the execution 
    runtime.generateDot("/tmp/getting-started.dot", true);
    
    // We generate an Svg trace of the execution
    runtime.generateTrace("/tmp/getting-started.svg");
    
    return 0;
}
