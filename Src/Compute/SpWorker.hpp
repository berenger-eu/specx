#ifndef SPWORKER_HPP
#define SPWORKER_HPP

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#ifdef SPETABARU_COMPILE_WITH_CUDA
#include "Cuda/SpCudaMemManager.hpp"
#endif
#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpAbstractTask.hpp"
#include "Utils/small_vector.hpp"

class SpComputeEngine;
class SpAbstractTaskGraph;

class SpWorker {
public:
    enum class SpWorkerType {
        CPU_WORKER,
#ifdef SPETABARU_COMPILE_WITH_CUDA
        GPU_WORKER,
#endif
        NB_WORKER_TYPES
    };
    
    static std::atomic<long int> totalNbThreadsCreated;
    

    static auto createDefaultTeamOfWorkers() {
#ifdef SPETABARU_COMPILE_WITH_CUDA
        return createHeterogeneousTeamOfWorkers();
#else
        return createTeamOfCpuWorkers();
#endif
    }

    static auto createTeamOfCpuWorkers(const int nbWorkers = SpUtils::DefaultNumThreads()) {
        small_vector<std::unique_ptr<SpWorker>> res;
        res.reserve(nbWorkers);
        
        for(int idxWorker = 0; idxWorker < nbWorkers; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerType::CPU_WORKER));
        }
        
        return res;
    }
    #ifdef SPETABARU_COMPILE_WITH_CUDA
    static auto createTeamOfGpuWorkers(const int nbWorkerPerGpus = SpCudaMemManager::getDefaultNbStreams(),
                                                 const int nbGpuWorkers = SpCudaMemManager::getNbDevices()) {
        small_vector<std::unique_ptr<SpWorker>> res;
        res.reserve(nbWorkerPerGpus*nbGpuWorkers);

        for(int idxGpu = 0; idxGpu < nbGpuWorkers; ++idxGpu) {
            for(int idxWorker = 0; idxWorker < nbWorkerPerGpus; ++idxWorker) {
                res.emplace_back(std::make_unique<SpWorker>(SpWorkerType::GPU_WORKER));
                res.back().gpuData.init(idxGpu);
            }
        }

        return res;
    }

    static auto createHeterogeneousTeamOfWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                                 const int nbWorkerPerGpus = SpCudaMemManager::getDefaultNbStreams(),
                                                 const int nbGpuWorkers = SpCudaMemManager::getNbDevices()) {
        small_vector<std::unique_ptr<SpWorker>> res;
        res.reserve(nbCpuWorkers + nbWorkerPerGpus*nbGpuWorkers);
        
        for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerType::CPU_WORKER));
        }

        for(int idxGpu = 0; idxGpu < nbGpuWorkers; ++idxGpu) {
            for(int idxWorker = 0; idxWorker < nbWorkerPerGpus; ++idxWorker) {
                res.emplace_back(std::make_unique<SpWorker>(SpWorkerType::GPU_WORKER));
                res.back().gpuData.init(idxGpu);
            }
        }
        
        return res;
    }
#endif
    
    static auto createDefaultWorkerTeam() {
        return createTeamOfWorkersOfType(SpUtils::DefaultNumThreads(), SpWorkerType::CPU_WORKER);
    }

    static void setWorkerForThread(SpWorker *w);
    static SpWorker* getWorkerForThread();

private:
    const SpWorkerType wt;
    std::mutex workerMutex;
    std::condition_variable workerConditionVariable;
    std::atomic<bool> stopFlag;
    std::atomic<SpComputeEngine*> ce;
    long int threadId;
    std::thread t;
#ifdef SPETABARU_COMPILE_WITH_CUDA
    SpCudaMemManager::WorkerData gpuData;
#endif

private:
    void setStopFlag(const bool inStopFlag) {
        stopFlag.store(inStopFlag, std::memory_order_relaxed);
    }
    
    bool hasBeenStopped() const {
        return stopFlag.load(std::memory_order_relaxed);
    }
    
    SpComputeEngine* getComputeEngine() const {
        return ce.load(std::memory_order_relaxed);
    }
    
    void execute(SpAbstractTask *task) {
        switch(this->getType()) {
            case SpWorkerType::CPU_WORKER:
                task->execute(SpCallableType::CPU);
                break;
                #ifdef SPETABARU_COMPILE_WITH_CUDA
            case SpWorkerType::GPU_WORKER:
                task->execute(SpCallableType::GPU);
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
        workerConditionVariable.wait(workerLock, [&]() { return stopFlag.load(std::memory_order_relaxed) || ce.load(std::memory_order_relaxed); });
    }
    
    void waitOnCe(SpComputeEngine* inCe, SpAbstractTaskGraph* atg);
    
    friend class SpComputeEngine;

public:

    explicit SpWorker(const SpWorkerType inWt) :
    wt(inWt), workerMutex(), workerConditionVariable(),
    stopFlag(false), ce(nullptr), threadId(0), t()
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
    
    SpWorkerType getType() const {
        return wt;
    }

#ifdef SPETABARU_COMPILE_WITH_CUDA
    SpCudaMemManager::WorkerData& getGpuData(){
        return gpuData;
    }
#endif
    
    void start();
    
    void doLoop(SpAbstractTaskGraph* inAtg);
};

#endif
