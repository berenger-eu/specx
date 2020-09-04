#ifndef SPABSTRACTTASKGRAPH_HPP
#define SPABSTRACTTASKGRAPH_HPP

#include "Schedulers/SpTasksManager.hpp"

class SpComputeEngine;
class SpAbstractTask;

class SpAbstractTaskGraph {
protected:
    //! Internal scheduler of tasks
    SpTasksManager scheduler;

public:
    void setComputeEngine(SpComputeEngine* inCe) {
        scheduler.setComputeEngine(inCe);
    }
    
    void preTaskExecution(SpAbstractTask* t) {
        scheduler.preTaskExecution(t);
    }
    
    void postTaskExecution(SpAbstractTask* t) {
        scheduler.postTaskExecution(t);
    }
    
    void waitAllTasks(){
        scheduler.waitAllTasks();
    }
    
    void waitRemain(const long int windowSize){
        scheduler.waitRemain(windowSize);
    }
    
};

#endif
