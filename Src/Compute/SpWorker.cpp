#include "Compute/SpWorker.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"
#include "Utils/SpHardware.hpp"

std::atomic<long int> SpWorker::totalNbThreadsCreated = 1;

thread_local SpWorker* workerForThread = nullptr;

void SpWorker::start() {
    if(!t.joinable()) {
        t = std::thread([&]() {
            SpUtils::SetThreadId(threadId);
            SpWorker::setWorkerForThread(this);
            
            doLoop(nullptr);
        });
    }
}

void SpWorker::waitOnCe(SpComputeEngine* inCe, SpAbstractTaskGraph* atg) {
    inCe->wait(*this, atg);
}

void SpWorker::setWorkerForThread(SpWorker *w) {
    workerForThread = w;
}

SpWorker* SpWorker::getWorkerForThread() {
    return workerForThread;
}

void SpWorker::doLoop(SpAbstractTaskGraph* inAtg) {
    while(!stopFlag.load(std::memory_order_relaxed) && (!inAtg || !inAtg->isFinished())) {
        SpComputeEngine* saveCe = nullptr;
        
        // Using memory order acquire on ce.load to form release/acquire pair
        // I think we could use memory order consume as all the code that follows depends on the load of ce (through saveCe).
        if((saveCe = ce.load(std::memory_order_acquire))) {
            if(saveCe->areThereAnyWorkersToMigrate()) {
                if(saveCe->areWorkersToMigrateOfType(wt)) {
                    auto previousNbOfWorkersToMigrate = saveCe->fetchDecNbOfWorkersToMigrate();
                    
                    if(previousNbOfWorkersToMigrate > 0) {
                        
                        SpComputeEngine* newCe = saveCe->getCeToMigrateTo();
                        ce.store(newCe, std::memory_order_relaxed);
                        
                        auto previousMigrationSignalingCounterVal = saveCe->fetchDecMigrationSignalingCounter();
                        
                        if(previousMigrationSignalingCounterVal == 1) {
                            saveCe->notifyMigrationFinished();
                        }
                        
                        continue;
                    }
                }
            }
            
            if(saveCe->areThereAnyReadyTasksForWorkerType(this->getType())){
                SpAbstractTask* task = saveCe->getTaskForWorkerType(this->getType());
                 
                if(task) {
                    SpAbstractTaskGraph* atg = task->getAbstractTaskGraph();
                    
                    if(this->getType() == SpWorker::SpWorkerType::GPU_WORKER) {
						std::unique_lock<std::mutex>(SpHardware::gpuMutexes[0 /* this->getGpuId() */]);
						atg->preTaskExecution(task, *this);
						execute(task);
						atg->postTaskExecution(task, *this);
					} else {
						atg->preTaskExecution(task, *this);
						execute(task);
						atg->postTaskExecution(task, *this);
					}
                    
                    continue;
                }
            }
            
            waitOnCe(saveCe, inAtg);
        } else {
            idleWait();
        }
        
    }
}



