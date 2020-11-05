#ifndef SPCOMPUTEENGINE_HPP
#define SPCOMPUTEENGINE_HPP

#include <memory>
#include <optional>
#include <utility>
#include <algorithm>
#include <iterator>
#include <atomic>

#include "Compute/SpWorker.hpp"
#include "Scheduler/SpPrioScheduler.hpp"
#include "Utils/small_vector.hpp"

class SpAbstractTaskGraph;

class SpComputeEngine {

private:
    small_vector<std::unique_ptr<SpWorker>> workers;
    std::mutex ceMutex;
    std::condition_variable ceCondVar;
    std::mutex migrationMutex;
    std::condition_variable migrationCondVar;
    SpPrioScheduler prioSched;
    std::atomic<long int> nbWorkersToMigrate;
    std::atomic<long int> migrationSignalingCounter;
    SpWorker::SpWorkerType workerTypeToMigrate;
    SpComputeEngine* ceToMigrateTo;
    long int nbAvailableCpuWorkers;
    long int nbAvailableGpuWorkers;
    long int totalNbCpuWorkers;
    long int totalNbGpuWorkers;
    bool hasBeenStopped;

private:
    
    auto sendWorkersToInternal(SpComputeEngine *otherCe, const SpWorker::SpWorkerType wt, const long int maxCount, const bool allowBusyWorkersToBeDetached) {
        small_vector<std::unique_ptr<SpWorker>> res;
        using iter_t = small_vector<std::unique_ptr<SpWorker>>::iterator;
        
        auto computeNbWorkersToDetach =
        [&]() {
            auto compute = 
            [](long int nbTotal, long int nbWaiting, const bool allowBusyWorToBeDeta, const long int max) {
                if(allowBusyWorToBeDeta) {
                    return std::min(nbTotal, max);
                } else {
                    return std::min(nbWaiting, max);
                }
            };
            switch(wt) {
                case SpWorker::SpWorkerType::CPU_WORKER:
                    return compute(totalNbCpuWorkers, nbAvailableCpuWorkers, allowBusyWorkersToBeDetached, maxCount);
                case SpWorker::SpWorkerType::GPU_WORKER:
                    return compute(totalNbGpuWorkers, nbAvailableGpuWorkers, allowBusyWorkersToBeDetached, maxCount);
                default:
                    return static_cast<long int>(0);
            }
        };
        
        const auto nbWorkersToDetach =
        [&]()    
        {
            std::unique_lock<std::mutex> computeEngineLock(ceMutex);
            
            auto result = computeNbWorkersToDetach();
            
            if(result > 0) {
                workerTypeToMigrate = wt;
                ceToMigrateTo = otherCe;
                migrationSignalingCounter.store(result, std::memory_order_relaxed);
                nbWorkersToMigrate.store(result, std::memory_order_release);
            }
            
            return result;
        }();
        
        if(nbWorkersToDetach > 0) {
            ceCondVar.notify_all();
            
            {
                std::unique_lock<std::mutex> migrationLock(migrationMutex);
                migrationCondVar.wait(migrationLock, [&](){ return !(migrationSignalingCounter.load(std::memory_order_acquire) > 0); });
            }
            
            auto startIt = std::move_iterator<iter_t>(workers.begin());
            auto endIt = std::move_iterator<iter_t>(workers.end());
            
            auto eraseStartPosIt = std::remove_if(startIt, endIt,
                                                [&](std::unique_ptr<SpWorker>&& wPtr) {
                                                    if(wPtr->getComputeEngine() != this) {
                                                        res.push_back(std::move(wPtr));
                                                        return true;
                                                    } else {
                                                        return false;
                                                    }
                                                });
            
            workers.erase(eraseStartPosIt.base(), workers.end());
            
            std::unique_lock<std::mutex> computeEngineLock(ceMutex);
            updateWorkerCounters<true, true>(wt, -nbWorkersToDetach);
        
        }
        
        return res;
    }
    
    template <const bool bindAndStartWorkers>
    void addWorkersInternal(small_vector_base<std::unique_ptr<SpWorker>>&& inWorkers) {
        for(auto& w : inWorkers) {
            updateWorkerCounters<true,false>(w->getType(), +1);
            if constexpr(bindAndStartWorkers) {
                w->bindTo(this);
                w->start();
            }
        }
        
        if(workers.empty()) {
            workers = std::move(inWorkers);
        } else {
            workers.reserve(workers.size() + inWorkers.size());
            std::move(std::begin(inWorkers), std::end(inWorkers), std::back_inserter(workers));
        }
    }
    
    bool areThereAnyWorkersToMigrate() const {
        return nbWorkersToMigrate.load(std::memory_order_acquire) > 0;
    }
    
    bool areThereAnyReadyTasks() const {
        return prioSched.getNbTasks() > 0;
    }
    
    bool areWorkersToMigrateOfType(SpWorker::SpWorkerType inWt) {
        return workerTypeToMigrate == inWt;
    }
    
    SpAbstractTask* getTask() {
        return prioSched.pop();
    }
    
    template <const bool updateTotalCounter, const bool updateAvailableCounter>
    void updateWorkerCounters(const SpWorker::SpWorkerType inWt, const long int addend) {
        switch(inWt) {
            case SpWorker::SpWorkerType::CPU_WORKER:
                if constexpr(updateTotalCounter) {
                    totalNbCpuWorkers += addend;
                }
                if constexpr(updateAvailableCounter) {
                    nbAvailableGpuWorkers += addend;
                }
                break;
            case SpWorker::SpWorkerType::GPU_WORKER:
                if constexpr(updateTotalCounter) {
                    totalNbGpuWorkers += addend;
                }
                
                if constexpr(updateAvailableCounter) {
                    nbAvailableGpuWorkers += addend;
                }
                break;
            default:
                break;
        }
    }
    
    void wait(SpWorker& worker, SpAbstractTaskGraph* atg);
    
    auto getCeToMigrateTo() {
        return ceToMigrateTo;
    }
    
    auto fetchDecNbOfWorkersToMigrate() {
        return nbWorkersToMigrate.fetch_sub(1, std::memory_order_relaxed);
    }
    
    void notifyMigrationFinished() {
        { 
            std::unique_lock<std::mutex> migrationLock(migrationMutex);
        }
        migrationCondVar.notify_one();
    }
    
    auto fetchDecMigrationSignalingCounter() {
        return migrationSignalingCounter.fetch_sub(1, std::memory_order_release);
    }
    
    friend void SpWorker::waitOnCe(SpComputeEngine* inCe, SpAbstractTaskGraph* atg);
    friend void SpWorker::doLoop(SpAbstractTaskGraph* atg);

public:
    explicit SpComputeEngine(small_vector_base<std::unique_ptr<SpWorker>>&& inWorkers)
    : workers(), ceMutex(), ceCondVar(), migrationMutex(), migrationCondVar(), prioSched(), nbWorkersToMigrate(0),
      migrationSignalingCounter(0),  workerTypeToMigrate(SpWorker::SpWorkerType::CPU_WORKER), ceToMigrateTo(nullptr), nbAvailableCpuWorkers(0),
      nbAvailableGpuWorkers(0), totalNbCpuWorkers(0), totalNbGpuWorkers(0), hasBeenStopped(false) {
        addWorkers(std::move(inWorkers));
    }
    
    explicit SpComputeEngine() : SpComputeEngine(small_vector<std::unique_ptr<SpWorker>, 0>{}) {}
    
    ~SpComputeEngine() {
        stopIfNotAlreadyStopped();
    }
    
    void pushTask(SpAbstractTask* t) {
        prioSched.push(t);
        wakeUpWaitingWorkers();
    }
    
    void pushTasks(small_vector_base<SpAbstractTask*>& tasks) {
        prioSched.pushTasks(tasks);
        wakeUpWaitingWorkers();
    }
    
    size_t getCurrentNbOfWorkers() const {
        return workers.size();
    }
    
    void addWorkers(small_vector_base<std::unique_ptr<SpWorker>>&& inWorkers) {
        addWorkersInternal<true>(std::move(inWorkers));
    }
    
    void sendWorkersTo(SpComputeEngine& otherCe, const SpWorker::SpWorkerType wt, const size_t maxCount, const bool allowBusyWorkersToBeDetached) {
        SpComputeEngine* otherCePtr = std::addressof(otherCe);
        
        if(otherCePtr && otherCePtr != this) {
            otherCePtr->addWorkersInternal<false>(sendWorkersToInternal(otherCePtr, wt, maxCount, allowBusyWorkersToBeDetached));
        }
    }
    
    auto detachWorkers(const SpWorker::SpWorkerType wt, const size_t maxCount, const bool allowBusyWorkersToBeDetached) {
        return sendWorkersToInternal(nullptr, wt, maxCount, allowBusyWorkersToBeDetached);
    }
    
    void stopIfNotAlreadyStopped();
    
    void wakeUpWaitingWorkers() {
        {
            std::unique_lock<std::mutex> ceLock(ceMutex);
        }
        ceCondVar.notify_all();
    }
};

#endif
