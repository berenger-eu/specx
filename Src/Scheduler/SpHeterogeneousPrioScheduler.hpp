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
#include "Data/SpDataAccessMode.hpp"

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
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > gpuTaskQueue;
    std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio > heterogeneousTaskQueue;

public:
    explicit SpHeterogeneousPrioScheduler() : mutexReadyTasks(), cpuTaskQueue(), gpuTaskQueue(), heterogeneousTaskQueue()
    {}

    // No copy or move
    SpHeterogeneousPrioScheduler(const SpHeterogeneousPrioScheduler&) = delete;
    SpHeterogeneousPrioScheduler(SpHeterogeneousPrioScheduler&&) = delete;
    SpHeterogeneousPrioScheduler& operator=(const SpHeterogeneousPrioScheduler&) = delete;
    SpHeterogeneousPrioScheduler& operator=(SpHeterogeneousPrioScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const{
        // TO DO : need to figure out how to get rid of this mutex lock
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            return static_cast<int>(cpuTaskQueue.size() + heterogeneousTaskQueue.size());
        }
        return static_cast<int>(gpuTaskQueue.size() + heterogeneousTaskQueue.size());
    }

    int push(SpAbstractTask* newTask){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        const bool hasCpuCallable = newTask->hasCallableOfType(SpCallableType::CPU);
        const bool hasGpuCallable = newTask->hasCallableOfType(SpCallableType::CUDA)
                || newTask->hasCallableOfType(SpCallableType::HIP);
        
        if(hasCpuCallable && hasGpuCallable) {
            heterogeneousTaskQueue.push(newTask);
        } else {
            if(hasCpuCallable) {
                cpuTaskQueue.push(newTask);
            } else {
                gpuTaskQueue.push(newTask);
            }
        }
        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks) {
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        
        for(auto t : tasks) {
            const bool hasCpuCallable = t->hasCallableOfType(SpCallableType::CPU);
            const bool hasGpuCallable = t->hasCallableOfType(SpCallableType::CUDA)
                    || t->hasCallableOfType(SpCallableType::HIP);
            
            if(hasCpuCallable && hasGpuCallable) {
                heterogeneousTaskQueue.push(t);
            } else {
                if(hasCpuCallable) {
                    cpuTaskQueue.push(t);
                } else {
                    gpuTaskQueue.push(t);
                }
            }
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt){
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        std::priority_queue<SpAbstractTask*, small_vector<SpAbstractTask*>, ComparePrio >* queue = nullptr;
        
        if(heterogeneousTaskQueue.size() > 0) {
            queue = std::addressof(heterogeneousTaskQueue);
        }
        
        if(wt == SpWorkerTypes::Type::CPU_WORKER && cpuTaskQueue.size() > 0) {
            SpAbstractTask* cpuTask = cpuTaskQueue.top();
            
            if(queue) {
                if(ComparePrio()(queue->top(), cpuTask)) {
                    queue = std::addressof(cpuTaskQueue);
                }
            } else {
                queue = std::addressof(cpuTaskQueue);
            }
        } else if(
          #ifdef SPECX_COMPILE_WITH_CUDA
                  wt == SpWorkerTypes::Type::CUDA_WORKER &&
          #endif
          #ifdef SPECX_COMPILE_WITH_HIP
                  wt == SpWorkerTypes::Type::HIP_WORKER &&
          #endif
                  gpuTaskQueue.size() > 0) {
            SpAbstractTask* cudaTask = gpuTaskQueue.top();
            
            if(queue) {
                if(ComparePrio()(queue->top(), cudaTask)) {
                    queue = std::addressof(gpuTaskQueue);
                }
            } else {
                queue = std::addressof(gpuTaskQueue);
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
