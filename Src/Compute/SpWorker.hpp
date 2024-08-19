#ifndef SPWORKER_HPP
#define SPWORKER_HPP

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <functional>

#include "Config/SpConfig.hpp"

#ifdef SPECX_COMPILE_WITH_CUDA
#include "Cuda/SpCudaWorkerData.hpp"
#include "Cuda/SpCudaMemManager.hpp"
#endif
#ifdef SPECX_COMPILE_WITH_HIP
#include "Hip/SpHipWorkerData.hpp"
#include "Hip/SpHipMemManager.hpp"
#endif
#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpAbstractTask.hpp"
#include "Utils/small_vector.hpp"
#include "SpWorkerTypes.hpp"

class SpComputeEngine;
class SpAbstractTaskGraph;
class SpWorkerTeamBuilder;

class SpWorker {
public:   
    static std::atomic<long int> totalNbThreadsCreated;

    static void setWorkerForThread(SpWorker *w);
    static SpWorker* getWorkerForThread();

private:
    const SpWorkerTypes::Type wt;
    std::mutex workerMutex;
    std::condition_variable workerConditionVariable;
    std::atomic<bool> stopFlag;
    std::atomic<SpComputeEngine*> ce;
    long int threadId;
    std::thread t;
#ifdef SPECX_COMPILE_WITH_CUDA
    SpCudaWorkerData cudaData;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
    SpHipWorkerData hipData;
#endif

    std::atomic<std::function<void(void)>*> hasFuncToExecPtr;

    void execFuncIfNeeded(){
        if(hasFuncToExecPtr.load(std::memory_order_relaxed)){
            SpDebugPrint() << "execFuncIfNeeded hasFuncToExec set.";
            (*hasFuncToExecPtr)();
            SpDebugPrint() << "execFuncIfNeeded done.";
            hasFuncToExecPtr = nullptr;
        }
    }

private:
    void setStopFlag(const bool inStopFlag) {
        stopFlag.store(inStopFlag, std::memory_order_relaxed);
    }
    
    bool hasBeenStopped() const {
        return stopFlag.load(std::memory_order_relaxed);
    }
    
    bool hasSpecificActionToDo() const {
        return hasFuncToExecPtr.load(std::memory_order_relaxed);
    }
    
    SpComputeEngine* getComputeEngine() const {
        return ce.load(std::memory_order_relaxed);
    }
    
    void execute(SpAbstractTask *task) {
        switch(this->getType()) {
            case SpWorkerTypes::Type::CPU_WORKER:
                task->execute(SpCallableType::CPU);
                break;
#ifdef SPECX_COMPILE_WITH_CUDA
            case SpWorkerTypes::Type::CUDA_WORKER:
                task->execute(SpCallableType::CUDA);
                break;
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            case SpWorkerTypes::Type::HIP_WORKER:
                task->execute(SpCallableType::HIP);
                break;
#endif
            default:
                assert(false && "Worker is of unknown type.");
        }
    }
    
    void waitForThread() {
        t.join();
    }
    
    void stop() {
        if(t.joinable()) {
            if(stopFlag.load(std::memory_order_relaxed)) {
                {
                    std::unique_lock<std::mutex> workerLock(workerMutex);
                    stopFlag.store(true, std::memory_order_relaxed);
                }
                workerConditionVariable.notify_one();
            }
            waitForThread();
        }
    }
    
    void bindTo(SpComputeEngine* inCe) {
        if(inCe) {
            {
                std::unique_lock workerLock(workerMutex);
                ce.store(inCe, std::memory_order_release);
            }
            workerConditionVariable.notify_one();
        }
    }
    
    void idleWait() {
        std::unique_lock<std::mutex> workerLock(workerMutex);
        workerConditionVariable.wait(workerLock, [&]() {
            return stopFlag.load(std::memory_order_relaxed)
                    || ce.load(std::memory_order_relaxed)
                    || hasFuncToExecPtr.load(std::memory_order_relaxed); });
    }
    
    void waitOnCe(SpComputeEngine* inCe, SpAbstractTaskGraph* atg);
    void wakeUpCe();
    
    friend class SpComputeEngine;

public:

    explicit SpWorker(const SpWorkerTypes::Type inWt) :
    wt(inWt), workerMutex(), workerConditionVariable(),
    stopFlag(false), ce(nullptr), threadId(0), t(), hasFuncToExecPtr(nullptr)
    {
        threadId = totalNbThreadsCreated.fetch_add(1, std::memory_order_relaxed);
    }

    SpWorker(const SpWorker& other) = delete;
    SpWorker(SpWorker&& other) = delete;
    SpWorker& operator=(const SpWorker& other) = delete;
    SpWorker& operator=(SpWorker&& other) = delete;
    
    ~SpWorker() {
        stop();
    }
    
    SpWorkerTypes::Type getType() const {
        return wt;
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    SpCudaWorkerData& getCudaData(){
        return cudaData;
    }
#endif

#ifdef SPECX_COMPILE_WITH_HIP
    SpHipWorkerData& getHipData(){
        return hipData;
    }
#endif
    
    void start();
    
    void doLoop(SpAbstractTaskGraph* inAtg);

    template <class ClassFunc>
    void setExecFunc(ClassFunc&& func) {
        assert(stopFlag == false);

        std::function<void(void)> funcToExec = [&func, id = threadId, type = wt](){
            func(id, type);
        };

        hasFuncToExecPtr.store(&funcToExec, std::memory_order_release);

        wakeUpCe();
        // Ce could be set but the worker still on the idleWait
        workerConditionVariable.notify_one();
        
        while(hasFuncToExecPtr.load(std::memory_order_relaxed)){
            usleep(100);
        }
    }

    friend SpWorkerTeamBuilder;
};

#endif
