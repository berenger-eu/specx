///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSCHEDULER_HPP
#define SPSCHEDULER_HPP

#include <atomic>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"


//! The runtime is the main component of specx.
class SpSimpleScheduler{
    std::atomic<SpAbstractTask*> tasksReady;
    std::atomic<int> nbReadyTasks;

public:
    explicit SpSimpleScheduler() : tasksReady(nullptr), nbReadyTasks(0) {
    }

    // No copy or move
    SpSimpleScheduler(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler(SpSimpleScheduler&&) = delete;
    SpSimpleScheduler& operator=(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler& operator=(SpSimpleScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const{
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            return nbReadyTasks;
        }

        return 0;
    }

    int push(SpAbstractTask* newTask){
        int cpt = nbReadyTasks;
        while(!nbReadyTasks.compare_exchange_strong(cpt, cpt+1)){
        }
        SpAbstractTask* expectedTask = tasksReady;
        newTask->getPtrNextList() = expectedTask;
        while(!tasksReady.compare_exchange_strong(expectedTask, newTask)){
            newTask->getPtrNextList() = expectedTask;
        }
        return 1;
    }

    int pushTasks(small_vector_base<SpAbstractTask*>& tasks) {
        for(auto tk : tasks){
            push(tk);
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt){
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            while(true){
                int ticket = nbReadyTasks;
                if(ticket == 0){
                    return nullptr;
                }
                if(nbReadyTasks.compare_exchange_strong(ticket, ticket-1)){
                    SpAbstractTask* task = tasksReady;
                    while(task == nullptr || !tasksReady.compare_exchange_strong(task, task->getPtrNextList())){
                        if(task == nullptr){
                            task = tasksReady;
                        }
                    }
                    task->getPtrNextList() = nullptr;
                    return task;
                }
            }
        }
        return nullptr;
    }
};


#endif
