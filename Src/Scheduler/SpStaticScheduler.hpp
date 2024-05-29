///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSTATICSCHEDULER_HPP
#define SPSTATICSCHEDULER_HPP

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

template <int MaxNbMemoryNodes = 128, const bool EnableStealTask = false>
class SpStaticScheduler : public SpAbstractScheduler{
    static_assert(MaxNbMemoryNodes > 0, "MaxNbMemoryNodes must be greater than 0");

    struct TaskWrapper{
        SpAbstractTask* task;
        int priority;
        int nodeId;
    };

    struct ComparePrio{
        bool operator()(const TaskWrapper& lhs, const TaskWrapper& rhs) const
        {
            return lhs.priority < rhs.priority;
        }
    };

    //! To protect the tasksReady list
    mutable std::mutex mutexReadyTasks[MaxNbMemoryNodes];
    
    std::priority_queue<TaskWrapper, small_vector<TaskWrapper>, ComparePrio > taskQueues[MaxNbMemoryNodes];

    std::atomic<int> nbTasks[int(SpWorkerTypes::Type::NB_WORKER_TYPES)];

public:
    explicit SpStaticScheduler() {}

    // No copy or move
    SpStaticScheduler(const SpStaticScheduler&) = delete;
    SpStaticScheduler(SpStaticScheduler&&) = delete;
    SpStaticScheduler& operator=(const SpStaticScheduler&) = delete;
    SpStaticScheduler& operator=(SpStaticScheduler&&) = delete;

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
        std::unique_lock<std::mutex> locker(mutexReadyTasks);
        const bool hasCpuCallable = newTask->hasCallableOfType(SpCallableType::CPU);
        const bool hasGpuCallable = newTask->hasCallableOfType(SpCallableType::CUDA)
                || newTask->hasCallableOfType(SpCallableType::HIP);
        
        if(hasCpuCallable) {
            mutexReadyTasks[int(SpWorkerTypes::Type::CPU_WORKER)] += 1;
        } 
        if(hasGpuCallable) {
#ifdef SPECX_COMPILE_WITH_CUDA
            mutexReadyTasks[int(SpWorkerTypes::Type::CUDA_WORKER)] += 1;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            mutexReadyTasks[int(SpWorkerTypes::Type::HIP_WORKER)] += 1;
#endif
        }

        TaskWrapper wrapper;
        wrapper.task = newTask;
        wrapper.priority = GetRealPriority(newTask->getPriority());
        wrapper.nodeId = GetTargetNode(newTask->getPriority());

        assert(wrapper.nodeId != -1);
        assert(wrapper.nodeId < MaxNbMemoryNodes);

        mutexReadyTasks[wrapper.nodeId].lock();

        taskQueues[wrapper.nodeId].push(wrapper);

        mutexReadyTasks[wrapper.nodeId].unlock();

        return 1;
    }
    
    int pushTasks(small_vector_base<SpAbstractTask*>& tasks)  final {        
        for(auto t : tasks) {
            push(t);
        }
        return int(tasks.size());
    }

    SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt) final {
        const int currentnodeId = SpUtils::GetDeviceId() + 1;
        assert(currentnodeId < MaxNbMemoryNodes);
        assert(currentnodeId >= 0);
        assert(SpUtils::GetThreadType() == wt);

        if constexpr(EnableStealTask == false) {
            mutexReadyTasks[currentnodeId].lock();
            if(taskQueues[currentnodeId].size()){
                SpAbstractTask* task = taskQueues[currentnodeId].top().task;
                taskQueues[currentnodeId].pop();
                mutexReadyTasks[currentnodeId].unlock();

                const bool hasCpuCallable = task->hasCallableOfType(SpCallableType::CPU);
                const bool hasGpuCallable = task->hasCallableOfType(SpCallableType::CUDA)
                        || task->hasCallableOfType(SpCallableType::HIP);
        
                if(hasCpuCallable) {
                    mutexReadyTasks[SpWorkerTypes::Type::CPU_WORKER] -= 1;
                } 
                if(hasGpuCallable) {
#ifdef SPECX_COMPILE_WITH_CUDA
                    mutexReadyTasks[SpWorkerTypes::Type::CUDA_WORKER] -= 1;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
                    mutexReadyTasks[SpWorkerTypes::Type::HIP_WORKER] -= 1;
#endif
                }
                return task;
            }
            return nullptr;
        }
        else{
            mutexReadyTasks[currentnodeId].lock();
            if(taskQueues[currentnodeId].size()){
                SpAbstractTask* task = taskQueues[currentnodeId].top().task;
                taskQueues[currentnodeId].pop();
                mutexReadyTasks[currentnodeId].unlock();

                const bool hasCpuCallable = task->hasCallableOfType(SpCallableType::CPU);
                const bool hasGpuCallable = task->hasCallableOfType(SpCallableType::CUDA)
                        || task->hasCallableOfType(SpCallableType::HIP);
        
                if(hasCpuCallable) {
                    mutexReadyTasks[SpWorkerTypes::Type::CPU_WORKER] -= 1;
                } 
                if(hasGpuCallable) {
#ifdef SPECX_COMPILE_WITH_CUDA
                    mutexReadyTasks[SpWorkerTypes::Type::CUDA_WORKER] -= 1;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
                    mutexReadyTasks[SpWorkerTypes::Type::HIP_WORKER] -= 1;
#endif
                }
            }
            mutexReadyTasks[currentnodeId].unlock();
            if(mutexReadyTasks[wt]){
                for(int idxNode = 0; idxNode < MaxNbMemoryNodes; ++idxNode){
                    const int idxCandidate = (currentnodeId + idxNode) % MaxNbMemoryNodes;
                    if(idxCandidate == currentnodeId){
                        continue;
                    }
                    mutexReadyTasks[idxCandidate].lock();
                    if(taskQueues[idxCandidate].size()){
                        SpAbstractTask* potentialTask = taskQueues[idxCandidate].top().task;

                        const bool hasCpuCallable = potentialTask->hasCallableOfType(SpCallableType::CPU);
                        const bool hasGpuCallable = potentialTask->hasCallableOfType(SpCallableType::CUDA)
                                || potentialTask->hasCallableOfType(SpCallableType::HIP);
                        const bool canTakeIt = (wt == SpWorkerTypes::Type::CPU_WORKER && hasCpuCallable)
    #ifdef SPECX_COMPILE_WITH_CUDA
                                || (wt == SpWorkerTypes::Type::CUDA_WORKER && hasGpuCallable)
    #endif
    #ifdef SPECX_COMPILE_WITH_HIP
                                || (wt == SpWorkerTypes::Type::HIP_WORKER && hasGpuCallable)
    #endif
                                ;
                        if(canTakeIt){
                            taskQueues[idxCandidate].pop();
                            mutexReadyTasks[idxCandidate].unlock();

                            if(hasCpuCallable) {
                                mutexReadyTasks[SpWorkerTypes::Type::CPU_WORKER] -= 1;
                            } 
                            if(hasGpuCallable) {
    #ifdef SPECX_COMPILE_WITH_CUDA
                                mutexReadyTasks[SpWorkerTypes::Type::CUDA_WORKER] -= 1;
    #endif
    #ifdef SPECX_COMPILE_WITH_HIP
                                mutexReadyTasks[SpWorkerTypes::Type::HIP_WORKER] -= 1;
    #endif
                            }
                            return potentialTask;
                        }
                    }
                    mutexReadyTasks[idxCandidate].unlock();
                }
            }
        }
        return nullptr;
    }

    static auto GetRealPriority(const int inPriority){
        const int bitsForWorker = sizeof(int) * CHAR_BIT - __builtin_clz(MaxNbMemoryNodes);
        return inPriority >> bitsForWorker;
    }

    static auto GetTargetNode(const int inPriority){
        const int bitsForWorker = sizeof(int) * CHAR_BIT - __builtin_clz(MaxNbMemoryNodes);
        const int workerMask = (1 << bitsForWorker) - 1;
        const int nodeId = inPriority & workerMask;
        assert(0 <= nodeId && nodeId < MaxNbMemoryNodes);
        return nodeId;
    }

    static auto GeneratePriorityWorkerPair(const int inPriority, const int inNode){
        assert(0 <= inNode+1 && inNode+1 < MaxNbMemoryNodes);
        const int bitsForWorker = sizeof(int) * CHAR_BIT - __builtin_clz(MaxNbMemoryNodes);
        return (inPriority << bitsForWorker) | (inNode+1);
    }
};


#endif
