#ifndef SPABSTRACTTASKGRAPH_HPP
#define SPABSTRACTTASKGRAPH_HPP

#include "Schedulers/SpTasksManager.hpp"
#include "Output/SpDotDag.hpp"
#include "Output/SpSvgTrace.hpp"

class SpComputeEngine;
class SpAbstractTask;

class SpAbstractTaskGraph {
protected:
    //! Creation time point
    SpTimePoint startingTime;
    
    //! Internal scheduler of tasks
    SpTasksManager scheduler;

protected:
    
    void preTaskExecution(SpAbstractTask* t) {
        scheduler.preTaskExecution(t);
    }
    
    void postTaskExecution(SpAbstractTask* t) {
        scheduler.postTaskExecution(t);
    }

    friend void SpWorker::doLoop(SpAbstractTaskGraph*);
    
public:
    void computeOn(SpComputeEngine& inCe) {
        scheduler.setComputeEngine(std::addressof(inCe));
    }
    
    void waitAllTasks(){
        scheduler.waitAllTasks();
    }
    
    void waitRemain(const long int windowSize){
        scheduler.waitRemain(windowSize);
    }
    
    void finish();
    
    bool isFinished() const {
        return scheduler.isFinished();
    }

    void generateDot(const std::string& outputFilename, bool printAccesses=false) const {
        SpDotDag::GenerateDot(outputFilename, scheduler.getFinishedTaskList(), printAccesses);
    }

    void generateTrace([[maybe_unused]] const std::string& outputFilename, [[maybe_unused]] const bool showDependences = true) const {
        const SpComputeEngine * ce = scheduler.getComputeEngine();
        
        if(ce) {
            SpSvgTrace::GenerateTrace(outputFilename, scheduler.getFinishedTaskList(), startingTime, showDependences);
        }
    }
    
};

#endif
