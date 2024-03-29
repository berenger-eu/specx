///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPPRIOSCHEDULER_HPP
#define SPPRIOSCHEDULER_HPP

#include <vector>
#include <queue>
#include <utility>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"
#include "SpAbstractScheduler.hpp"

class SpPrioScheduler : public SpAbstractScheduler{
    struct ComparePrio{
        bool operator()(const SpAbstractTask* lhs, const SpAbstractTask* rhs) const
        {
            return lhs->getPriority() < rhs->getPriority();
        }
    };

    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasks;
    //! Contains the tasks that can be executed
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > tasksReady;
    
    std::atomic<int> nbReadyTasks;

public:
    explicit SpPrioScheduler() : mutexReadyTasks(), tasksReady(), nbReadyTasks(0) {
    }

    // No copy or move
    SpPrioScheduler(const SpPrioScheduler&) = delete;
    SpPrioScheduler(SpPrioScheduler&&) = delete;
    SpPrioScheduler& operator=(const SpPrioScheduler&) = delete;
    SpPrioScheduler& operator=(SpPrioScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const final {
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            return nbReadyTasks;
        }
        
        return 0;
    }

    int push(SpAbstractTask* newTask) final {
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        nbReadyTasks++;
        tasksReady.push(newTask);
        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks)  final {
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        nbReadyTasks += int(tasks.size());
        for(auto t : tasks) {
            tasksReady.push(t);
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt) final {
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            std::unique_lock<std::mutex> locker(mutexReadyTasks);
            if(tasksReady.size()){
                nbReadyTasks--;
                auto res = tasksReady.top();
                tasksReady.pop();
                return res;
            }
            return nullptr;
        }
        return nullptr;
    }
};


#endif
