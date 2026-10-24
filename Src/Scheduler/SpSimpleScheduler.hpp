///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSCHEDULER_HPP
#define SPSCHEDULER_HPP

#include <atomic>
#include <mutex>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"
#include "SpAbstractScheduler.hpp"


//! The runtime is the main component of specx.
class SpSimpleScheduler : public SpAbstractScheduler{
    // Serialize node access: a popped task can be destroyed with its graph
    // while another consumer is still inspecting the old head of the list.
    std::mutex mutexReadyTasks;
    SpAbstractTask* tasksReady;
    std::atomic<int> nbReadyTasks;

public:
    explicit SpSimpleScheduler() : tasksReady(nullptr), nbReadyTasks(0) {
    }

    // No copy or move
    SpSimpleScheduler(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler(SpSimpleScheduler&&) = delete;
    SpSimpleScheduler& operator=(const SpSimpleScheduler&) = delete;
    SpSimpleScheduler& operator=(SpSimpleScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const final {
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            return nbReadyTasks;
        }

        return 0;
    }

    int push(SpAbstractTask* newTask) final {
        std::lock_guard<std::mutex> lock(mutexReadyTasks);
        newTask->getPtrNextList() = tasksReady;
        tasksReady = newTask;
        ++nbReadyTasks;
        return 1;
    }

    int pushTasks(small_vector_base<SpAbstractTask*>& tasks) final {
        std::lock_guard<std::mutex> lock(mutexReadyTasks);
        for(auto* task : tasks) {
            task->getPtrNextList() = tasksReady;
            tasksReady = task;
        }
        nbReadyTasks += int(tasks.size());
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt) final {
        if(wt != SpWorkerTypes::Type::CPU_WORKER) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutexReadyTasks);
        if(!tasksReady) {
            return nullptr;
        }
        auto* task = tasksReady;
        tasksReady = task->getPtrNextList();
        task->getPtrNextList() = nullptr;
        --nbReadyTasks;
        return task;
    }

};


#endif
