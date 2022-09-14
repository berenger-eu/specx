///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSCHEDULER_HPP
#define SPSCHEDULER_HPP

#include <list>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"


//! The runtime is the main component of specx.
class SpSimpleScheduler{
    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasks;
    //! Contains the tasks that can be executed
    std::vector<SpAbstractTask*> tasksReady;
    
    std::atomic<int> nbReadyTasks;

public:
    explicit SpSimpleScheduler() : mutexReadyTasks(), tasksReady(), nbReadyTasks(0) {
        tasksReady.resize(128);
    }

    // No copy or move
    SpSimpleScheduler(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler(SpSimpleScheduler&&) = delete;
    SpSimpleScheduler& operator=(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler& operator=(SpSimpleScheduler&&) = delete;

    int getNbTasks() const{
        return nbReadyTasks;
    }

    int push(SpAbstractTask* newTask){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        nbReadyTasks++;
        tasksReady.push_back(newTask);
        return 1;
    }

    SpAbstractTask* pop(){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        if(tasksReady.size()){
            nbReadyTasks--;
            SpAbstractTask* newTask = tasksReady.back();
            tasksReady.pop_back();
            return newTask;
        }
        else{
            return nullptr;
        }
    }
};


#endif
