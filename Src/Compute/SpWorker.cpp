#include "Compute/SpWorker.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"

std::atomic<long int> SpWorker::totalNbThreadsCreated = 1;

void SpWorker::start() {
    if(!t.joinable()) {
        t = std::thread([&]() {
            SpUtils::SetThreadId(threadId);
            
            while(!stopFlag.load(std::memory_order_relaxed)) {
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
                    
                    if(saveCe->areThereAnyReadyTasks()){
                        SpAbstractTask* task = saveCe->getTask();
                         
                        if(task) {
                            SpAbstractTaskGraph* atg = task->getAbstractTaskGraph();
                            
                            atg->preTaskExecution(task);
                            
                            execute(task);
                            
                            atg->postTaskExecution(task);
                            
                            continue;
                        }
                    }
                    
                    waitOnCe(saveCe);
                } else {
                    idleWait();
                }
                
            }
        });
    }
}

void SpWorker::waitOnCe(SpComputeEngine* inCe) {
    inCe->wait(*this);
}
