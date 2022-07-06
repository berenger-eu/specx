///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

int main(){
    std::cout << "Example basis:" << std::endl;
    {
        // Create the runtime
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        const int initVal = 1;
        int writeVal = 0;
        // Create a task with lambda function
        runtime.task(SpRead(initVal), SpWrite(writeVal),
                     [](const int& initValParam, int& writeValParam){
            writeValParam += initValParam;
        });
        // Create a task with lambda function (that returns a bool)
        auto returnValue = runtime.task(SpRead(initVal), SpWrite(writeVal),
                     [](const int& initValParam, int& writeValParam) -> bool {
            writeValParam += initValParam;
            return true;
        });
        // Wait completion of a single task
        returnValue.wait();
        // Get the value of the task
        const bool res = returnValue.getValue();
        (void)res;
        // Wait until two tasks (or less) remain
        runtime.waitRemain(2);
        // Wait for all tasks to be done
        runtime.waitAllTasks();
        // Save trace and .dot
        runtime.generateTrace("/tmp/basis-trace.svg");
        runtime.generateDot("/tmp/basis-dag.dot");
    }
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

        runtime.task(SpRead(val), [](const int& /*valParam*/){
        }).setTaskName("A");

        runtime.task(SpRead(val), [&promise1](const int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("B");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("C");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), [&counterFirstSpec](int& valParam){
            valParam += 1;
            counterFirstSpec += 1;
        }).setTaskName("D");
        // val is 1


        runtime.task(SpPotentialWrite(val), [](int& valParam) -> bool {
            valParam += 2;
            return true;
        }).setTaskName("E");
        // val is 3

        std::atomic_int counterSecondSpec(0);

        runtime.task(SpWrite(val), [&counterSecondSpec](int& valParam){
            valParam *= 2;
            counterSecondSpec += 1;
        }).setTaskName("F");
        // val is 6

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex1-trace.svg");
        runtime.generateDot("/tmp/ex1-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 1
        std::cout << "counterSecondSpec = " << counterSecondSpec << std::endl; // 2
    }
    std::cout << "Example 2:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpRead(val), [](const int& /*valParam*/){
        }).setTaskName("A");

        runtime.task(SpWrite(val), [&promise1](int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("B");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("C");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), [&counterFirstSpec](int& valParam){
            valParam += 1;
            counterFirstSpec += 1;
        }).setTaskName("D");
        // val is 1


        runtime.task(SpPotentialWrite(val), [](int& valParam) -> bool {
            valParam += 2;
            return true;
        }).setTaskName("E");
        // val is 3

        std::atomic_int counterSecondSpec(0);

        runtime.task(SpWrite(val), [&counterSecondSpec](int& valParam){
            valParam *= 2;
            counterSecondSpec += 1;
        }).setTaskName("F");
        // val is 6

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex2-trace.svg");
        runtime.generateDot("/tmp/ex2-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 1
        std::cout << "counterSecondSpec = " << counterSecondSpec << std::endl; // 2
    }
    std::cout << "Example 3:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpWrite(val), [&promise1](int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("A");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("B");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpRead(val), [&counterFirstSpec](const int& /*valParam*/){
            counterFirstSpec += 1;
        }).setTaskName("C");

        runtime.task(SpPotentialWrite(val), [](int& valParam) -> bool {
            valParam += 2;
            return true;
        }).setTaskName("D");
        // val is 2

        std::atomic_int counterSecondSpec(0);

        runtime.task(SpRead(val), [&counterSecondSpec](const int& /*valParam*/){
            counterSecondSpec += 1;
        }).setTaskName("E");
        // val is 2

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex3-trace.svg");
        runtime.generateDot("/tmp/ex3-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 1
        std::cout << "counterSecondSpec = " << counterSecondSpec << std::endl; // 2
    }
    std::cout << "Example 4:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpWrite(val), [&promise1](int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("A");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("B");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("C");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), [&counterFirstSpec](int& valParam){
            valParam += 1;
            counterFirstSpec += 1;
        }).setTaskName("D");
        // val is 1

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex4-trace.svg");
        runtime.generateDot("/tmp/ex4-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 2
    }
    std::cout << "Example 5:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        std::promise<int> promise1;

        runtime.task(SpWrite(val), [&promise1](int& /*valParam*/){
            promise1.get_future().get();
        }).setTaskName("A");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("B");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return true;
        }).setTaskName("C");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), [&counterFirstSpec](int& valParam){
            valParam += 1;
            counterFirstSpec += 1;
        }).setTaskName("D");
        // val is 1

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex5-trace.svg");
        runtime.generateDot("/tmp/ex5-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 2
    }
    std::cout << "Example 6:" << std::endl;
    {
        SpRuntime runtime;

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/,
                                   const SpProbability& /*inProbability*/) -> bool{
            // Always speculate
            return true;
        });

        int val = 0;
        int val2 = 0;
        int val3 = 0;
        std::promise<int> promise1;

        runtime.task(SpWrite(val), SpWrite(val2), SpWrite(val3),
                     [&promise1](int& /*valParam*/, int& /*valParam2*/, int& /*valParam3*/){
            promise1.get_future().get();
        }).setTaskName("A");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return false;
        }).setTaskName("B");

        runtime.task(SpPotentialWrite(val), [](int& /*valParam*/) -> bool {
            return true;
        }).setTaskName("C");

        runtime.task(SpPotentialWrite(val2), [](int& /*valParam2*/) -> bool {
            return true;
        }).setTaskName("D");

        runtime.task(SpPotentialWrite(val3), [](int& /*valParam3*/) -> bool {
            return true;
        }).setTaskName("E");

        std::atomic_int counterFirstSpec(0);

        runtime.task(SpWrite(val), SpWrite(val2), SpRead(val3),
                     [&counterFirstSpec](int& valParam, int& valParam2, const int& /*valParam3*/){
            valParam += 1;
            valParam2 += 1;
            counterFirstSpec += 1;
        }).setTaskName("F");


        std::atomic_int counterSecondSpec(0);

        runtime.task(SpRead(val3),
                     [&counterSecondSpec](const int& /*valParam3*/){
            counterSecondSpec += 1;
        }).setTaskName("G");

        promise1.set_value(0);

        runtime.waitAllTasks();

        runtime.generateTrace("/tmp/ex6-trace.svg");
        runtime.generateDot("/tmp/ex6-dag.dot");

        std::cout << "counterFirstSpec = " << counterFirstSpec << std::endl; // 2
        std::cout << "counterSecondSpec = " << counterSecondSpec << std::endl; // 2
    }
    return 0;
}


