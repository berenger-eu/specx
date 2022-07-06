///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPHETEROGENEOUSPRIOSCHEDULER_HPP
#define SPHETEROGENEOUSPRIOSCHEDULER_HPP

#include <vector>
#include <queue>
#include <utility>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"

class SpHeterogeneousPrioScheduler{
    struct ComparePrio{
        bool operator()(const SpAbstractTask* lhs, const SpAbstractTask* rhs) const
        {
            return lhs->getPriority() < rhs->getPriority();
        }
    };

    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasks;
    
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > cpuTaskQueue;
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > cudaTaskQueue;
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > heterogeneousTaskQueue;

public:
    explicit SpHeterogeneousPrioScheduler() : mutexReadyTasks(), cpuTaskQueue(), cudaTaskQueue(), heterogeneousTaskQueue()
    {}

    // No copy or move
    SpHeterogeneousPrioScheduler(const SpHeterogeneousPrioScheduler&) = delete;
    SpHeterogeneousPrioScheduler(SpHeterogeneousPrioScheduler&&) = delete;
    SpHeterogeneousPrioScheduler& operator=(const SpHeterogeneousPrioScheduler&) = delete;
    SpHeterogeneousPrioScheduler& operator=(SpHeterogeneousPrioScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorker::SpWorkerType wt) const{
        // TO DO : need to figure out how to get rid of this mutex lock
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        if(wt == SpWorker::SpWorkerType::CPU_WORKER) {
            return static_cast<int>(cpuTaskQueue.size() + heterogeneousTaskQueue.size());
        }
        return static_cast<int>(cudaTaskQueue.size() + heterogeneousTaskQueue.size());
    }

    int push(SpAbstractTask* newTask){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        const bool hasCpuCallable = newTask->hasCallableOfType(SpCallableType::CPU);
        const bool hasCudaCallable = newTask->hasCallableOfType(SpCallableType::CUDA);
        
        if(hasCpuCallable && hasCudaCallable) {
            heterogeneousTaskQueue.push(newTask);
        } else {
            if(hasCpuCallable) {
                cpuTaskQueue.push(newTask);
            } else {
                cudaTaskQueue.push(newTask);
            }
        }
        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks) {
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        
        for(auto t : tasks) {
            const bool hasCpuCallable = t->hasCallableOfType(SpCallableType::CPU);
            const bool hasCudaCallable = t->hasCallableOfType(SpCallableType::CUDA);
            
            if(hasCpuCallable && hasCudaCallable) {
                heterogeneousTaskQueue.push(t);
            } else {
                if(hasCpuCallable) {
                    cpuTaskQueue.push(t);
                } else {
                    cudaTaskQueue.push(t);
                }
            }
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorker::SpWorkerType wt){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio >* queue = nullptr;
        
        if(heterogeneousTaskQueue.size() > 0) {
            queue = std::addressof(heterogeneousTaskQueue);
        }
        
        if(wt == SpWorker::SpWorkerType::CPU_WORKER && cpuTaskQueue.size() > 0) {
            SpAbstractTask* cpuTask = cpuTaskQueue.top();
            
            if(queue) {
                if(ComparePrio()(queue->top(), cpuTask)) {
                    queue = std::addressof(cpuTaskQueue);
                }
            } else {
                queue = std::addressof(cpuTaskQueue);
            }
        } else if(wt == SpWorker::SpWorkerType::CUDA_WORKER && cudaTaskQueue.size() > 0) {
            SpAbstractTask* cudaTask = cudaTaskQueue.top();
            
            if(queue) {
                if(ComparePrio()(queue->top(), cudaTask)) {
                    queue = std::addressof(cudaTaskQueue);
                }
            } else {
                queue = std::addressof(cudaTaskQueue);
            }
        }
        
        if(queue) {
            SpAbstractTask * res = queue->top();
            
            queue->pop();
            
            return res;
        }
        
        return nullptr;
    }
};


#endif
