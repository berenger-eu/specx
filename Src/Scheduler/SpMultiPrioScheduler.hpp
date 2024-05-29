///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPMULTIPRIOSCHEDULER_HPP
#define SPMULTIPRIOSCHEDULER_HPP

#include <vector>
#include <queue>
#include <utility>

#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"
#include "Data/SpDataAccessMode.hpp"
#include "SpAbstractScheduler.hpp"

template <int MaxNbDevices = 8, const bool FavorLocality = true>
class SpMultiPrioScheduler : public SpAbstractScheduler{
    static_assert(MaxNbDevices > 0, "MaxNbDevices must be greater than 0");

    struct TaskWrapper{
        SpAbstractTask* task;
        int priority;
        std::shared_ptr<std::atomic<bool>> taken;
    };

    struct ComparePrio{
        bool operator()(const TaskWrapper& lhs, const TaskWrapper& rhs) const
        {
            return lhs.priority < rhs.priority;
        }
    };

    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasksGPU[MaxNbDevices];    
    std::priority_queue<TaskWrapper, small_vector<TaskWrapper>, ComparePrio > taskQueuesGPU[MaxNbDevices];

    mutable std::mutex mutexReadyTasksCPU;    
    std::priority_queue<TaskWrapper, small_vector<TaskWrapper>, ComparePrio > taskQueuesCPU;

    std::atomic<int> nbTasks[int(SpWorkerTypes::Type::NB_WORKER_TYPES)];

    void decrTaskCounter(SpAbstractTask* task){
        const bool hasCpuCallable = task->hasCallableOfType(SpCallableType::CPU);
        const bool hasGpuCallable = task->hasCallableOfType(SpCallableType::CUDA)
                || task->hasCallableOfType(SpCallableType::HIP);

        if(hasCpuCallable) {
            nbTasks[int(SpWorkerTypes::Type::CPU_WORKER)] -= 1;
        } 
        if(hasGpuCallable) {
#ifdef SPECX_COMPILE_WITH_CUDA
            nbTasks[int(SpWorkerTypes::Type::CUDA_WORKER)] -= 1;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            nbTasks[int(SpWorkerTypes::Type::HIP_WORKER)] -= 1;
#endif
        }
    }

public:
    explicit SpMultiPrioScheduler() {}

    // No copy or move
    SpMultiPrioScheduler(const SpMultiPrioScheduler&) = delete;
    SpMultiPrioScheduler(SpMultiPrioScheduler&&) = delete;
    SpMultiPrioScheduler& operator=(const SpMultiPrioScheduler&) = delete;
    SpMultiPrioScheduler& operator=(SpMultiPrioScheduler&&) = delete;

    int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const final {
        if(wt == SpWorkerTypes::Type::CPU_WORKER) {
            return nbTasks[int(SpWorkerTypes::Type::CPU_WORKER)];
        }
#ifdef SPECX_COMPILE_WITH_CUDA
        return nbTasks[int(SpWorkerTypes::Type::CUDA_WORKER)];
#endif
#ifdef SPECX_COMPILE_WITH_HIP
        return nbTasks[int(SpWorkerTypes::Type::HIP_WORKER)];
#endif
        assert(false && "Should not be here!");
        return -1;
    }

    int push(SpAbstractTask* newTask) final {
        const bool hasCpuCallable = newTask->hasCallableOfType(SpCallableType::CPU);
        const bool hasGpuCallable = newTask->hasCallableOfType(SpCallableType::CUDA)
                || newTask->hasCallableOfType(SpCallableType::HIP);
        
        if(hasCpuCallable) {
            nbTasks[int(SpWorkerTypes::Type::CPU_WORKER)] += 1;
        } 
        if(hasGpuCallable) {
#ifdef SPECX_COMPILE_WITH_CUDA
            nbTasks[int(SpWorkerTypes::Type::CUDA_WORKER)] += 1;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            nbTasks[int(SpWorkerTypes::Type::HIP_WORKER)] += 1;
#endif
        }

        std::shared_ptr<std::atomic<bool>> taken = std::make_shared<std::atomic<bool>>(false);

        TaskWrapper wrapper;
        wrapper.task = newTask;
        wrapper.taken = taken;

        if(hasCpuCallable) {
            mutexReadyTasksCPU.lock();
            wrapper.priority = GetCPUPriority(newTask->getPriority());
            taskQueuesCPU.push(wrapper);
            mutexReadyTasksCPU.unlock();
        }
        if(hasGpuCallable) {
            for(int idxDevice = 0 ; idxDevice < MaxNbDevices; ++idxDevice){
                wrapper.priority = GetGPUPriority(newTask->getPriority(), idxDevice);
                mutexReadyTasksGPU[idxDevice].lock();
                taskQueuesGPU[idxDevice].push(wrapper);
                mutexReadyTasksGPU[idxDevice].unlock();
            }
        }

        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks)  final {        
        for(auto t : tasks) {
            push(t);
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt) final {
        const int currentWorkerId = SpUtils::GetThreadId();
        assert(currentWorkerId < MaxNbDevices);
        assert(currentWorkerId >= 0);
        assert(SpUtils::GetThreadType() == wt);
        const int deviceId = SpUtils::GetDeviceId();
        assert(deviceId < MaxNbDevices);
        assert(deviceId == -1 || wt != SpWorkerTypes::Type::CPU_WORKER);

        if(deviceId == -1){
            mutexReadyTasksCPU.lock();
            while(taskQueuesCPU.size()){
                TaskWrapper taskWrapper = taskQueuesCPU.top();
                taskQueuesCPU.pop();
                if(taskWrapper.taken->exchange(true) == false){
                    mutexReadyTasksCPU.unlock();
                    decrTaskCounter(taskWrapper.task);
                    return taskWrapper.task;
                }
            }
            mutexReadyTasksCPU.unlock();
        }
        else{
            mutexReadyTasksGPU[deviceId].lock();
            while(taskQueuesGPU[deviceId].size()){
                TaskWrapper taskWrapper = taskQueuesGPU[deviceId].top();
                taskQueuesGPU[deviceId].pop();
                if(taskWrapper.taken->exchange(true) == false){
                    mutexReadyTasksGPU[deviceId].unlock();
                    decrTaskCounter(taskWrapper.task);
                    return taskWrapper.task;
                }
            }
            mutexReadyTasksGPU[deviceId].unlock();
        }
        return nullptr;
    }

    static auto GetCPUPriority(const int inPriority){
        const int bitsForGpuPrio = sizeof(int) * CHAR_BIT/2;
        const int cpuPrio = (inPriority >> bitsForGpuPrio);
        if(FavorLocality && SpUtils::GetDeviceId() == -1){
            return cpuPrio + (1 << (bitsForGpuPrio - 1));
        }
        return cpuPrio;
    }

    static auto GetGPUPriority(const int inPriority, const int inDeviceId){
        const int bitsForGpuPrio = sizeof(int) * CHAR_BIT/2;
        const int prioMask = (1 << bitsForGpuPrio) - 1;
        const int gpuPrio = inPriority & prioMask;
        if(FavorLocality && SpUtils::GetDeviceId() == inDeviceId){
            return gpuPrio + (1 << (bitsForGpuPrio - 1));
        }
        return gpuPrio;
    }

    static auto GeneratePriorityWorkerPair(const int inCpuPriority, const int inGpuPriority){
        const int bitsForGpuPrio = sizeof(int) * CHAR_BIT/2;
        return (inCpuPriority << bitsForGpuPrio) | inGpuPriority;
    }
};


#endif
