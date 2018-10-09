///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under MIT Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSCHEDULER_HPP
#define SPSCHEDULER_HPP

#include <list>

#include "Tasks/SpAbstractTask.hpp"
#include "Utils/SpPriority.hpp"


//! The runtime is the main component of spetabaru.
class SpSimpleScheduler{
    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasks;
    //! Contains the tasks that can be executed
    std::list<SpAbstractTask*> tasksReady;

public:
    explicit SpSimpleScheduler(){
    }

    // No copy or move
    SpSimpleScheduler(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler(SpSimpleScheduler&&) = delete;
    SpSimpleScheduler& operator=(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler& operator=(SpSimpleScheduler&&) = delete;

    int getNbTasks() const{
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        return static_cast<int>(tasksReady.size());
    }

    int push(SpAbstractTask* newTask){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        tasksReady.push_back(newTask);
        return 1;
    }

    SpAbstractTask* pop(){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        if(tasksReady.size()){
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
