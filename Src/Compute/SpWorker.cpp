#include "Compute/SpWorker.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"

std::atomic<long int> SpWorker::totalNbThreadsCreated = 1;

thread_local SpWorker* workerForThread = nullptr;

void SpWorker::start() {
    if(!t.joinable()) {
        t = std::thread([&]() {
            SpUtils::SetThreadId(threadId);
            SpWorker::setWorkerForThread(this);
#ifdef SPECX_COMPILE_WITH_CUDA
            if(this->getType() == SpWorkerType::CUDA_WORKER){
                cudaData.initByWorker();
            }
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            if(this->getType() == SpWorkerType::HIP_WORKER){
                hipData.initByWorker();
            }
#endif
            
            doLoop(nullptr);

#ifdef SPECX_COMPILE_WITH_CUDA
            if(this->getType() == SpWorkerType::CUDA_WORKER){
                cudaData.destroyByWorker();
            }
#endif
#ifdef SPECX_COMPILE_WITH_HIP
            if(this->getType() == SpWorkerType::HIP_WORKER){
                hipData.destroyByWorker();
            }
#endif
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
    const long int nbNullTasksBeforeWait = 10000000;
    long int nullTasksCounter = 0;

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
                    auto workerType = this->getType();
                    if(workerType == SpWorker::SpWorkerType::CPU_WORKER){
                        atg->preTaskExecution(task, *this);
                        execute(task);
                        atg->postTaskExecution(task, *this);
                    }
                    #ifdef SPECX_COMPILE_WITH_CUDA
                    else if(workerType == SpWorker::SpWorkerType::CUDA_WORKER) {
						atg->preTaskExecution(task, *this);
						execute(task);
						atg->postTaskExecution(task, *this);
                    }
#endif
#ifdef SPECX_COMPILE_WITH_HIP
                    else if(workerType == SpWorker::SpWorkerType::HIP_WORKER) {
                        atg->preTaskExecution(task, *this);
                        execute(task);
                        atg->postTaskExecution(task, *this);
                    }
#endif
                    else {
                        assert(0);
                    }
                    
                    nullTasksCounter = 0;
                    continue;
                }
            }
            
            if(nullTasksCounter == nbNullTasksBeforeWait){
                waitOnCe(saveCe, inAtg);
            }
            else{
                nullTasksCounter += 1;
            }
        } else {
            idleWait();
        }
        
    }
}



