#include <optional>
#include <mutex>
#include <algorithm>

#include "SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"
#include "Task/SpAbstractTask.hpp"
#include "SpWorker.hpp"

void SpComputeEngine::stopIfNotAlreadyStopped() {
    if(!hasBeenStopped) {
        {
            std::unique_lock<std::mutex> computeEngineLock(ceMutex);
            for(auto& w : workers) {
                w->setStopFlag(true);
            }
        }
        
        ceCondVar.notify_all();
        
        for(auto& w : workers) {
            w->waitForThread();
        }
        
        hasBeenStopped = true;
    }
}

void SpComputeEngine::wait(SpWorker& worker, SpAbstractTaskGraph* atg) {
    nbWaitingWorkers += 1;
    std::unique_lock<std::mutex> ceLock(ceMutex);
    updateWorkerCounters<false, true>(worker.getType(), +1);
    ceCondVar.wait(ceLock, [&]() { return worker.hasBeenStopped()
                || areThereAnyReadyTasksForWorkerType(worker.getType())
                || (atg && atg->isFinished())
                || areThereAnyWorkersToMigrate();});
    updateWorkerCounters<false, true>(worker.getType(), -1);
    nbWaitingWorkers -= 1;
}
