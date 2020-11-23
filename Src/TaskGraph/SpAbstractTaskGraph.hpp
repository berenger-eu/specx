#ifndef SPABSTRACTTASKGRAPH_HPP
#define SPABSTRACTTASKGRAPH_HPP

#include "Scheduler/SpTaskManager.hpp"
#include "Output/SpDotDag.hpp"
#include "Output/SpSvgTrace.hpp"

class SpComputeEngine;
class SpAbstractTask;
class SpWorker;

class SpAbstractTaskGraph {
protected:
    //! Creation time point
    SpTimePoint startingTime;
    
    //! Internal scheduler of tasks
    SpTaskManager scheduler;

protected:
    
    void preTaskExecution(SpAbstractTask* t, SpWorker& w) {
        scheduler.preTaskExecution(t, w);
    }
    
    void postTaskExecution(SpAbstractTask* t, SpWorker& w) {
        scheduler.postTaskExecution(t, w);
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
