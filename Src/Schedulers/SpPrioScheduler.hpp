///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPPRIOSCHEDULER_HPP
#define SPPRIOSCHEDULER_HPP

#include <vector>
#include <queue>
#include <utility>

#include "Tasks/SpAbstractTask.hpp"
#include "Utils/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"

class SpPrioScheduler{
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


public:
    explicit SpPrioScheduler() {
    }

    // No copy or move
    SpPrioScheduler(const SpPrioScheduler&) = delete;
    SpPrioScheduler(SpPrioScheduler&&) = delete;
    SpPrioScheduler& operator=(const SpPrioScheduler&) = delete;
    SpPrioScheduler& operator=(SpPrioScheduler&&) = delete;

    int getNbTasks() const{
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        return int(tasksReady.size());
    }

    int push(SpAbstractTask* newTask){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        tasksReady.push(newTask);
        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks) {
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        for(auto t : tasks) {
            tasksReady.push(t);
        }
        return int(tasks.size());
    }

    SpAbstractTask* pop(){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        if(tasksReady.size()){
            auto res = tasksReady.top();
            tasksReady.pop();
            return res;
        }
        return nullptr;
    }
};


#endif
