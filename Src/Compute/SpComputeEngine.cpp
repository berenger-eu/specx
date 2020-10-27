#include <optional>
#include <mutex>
#include <algorithm>

#include "SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"
#include "Tasks/SpAbstractTask.hpp"
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
    std::unique_lock<std::mutex> ceLock(ceMutex);
    updateWorkerCounters<false, true>(worker.getType(), +1);
    ceCondVar.wait(ceLock, [&]() { return worker.hasBeenStopped() || areThereAnyWorkersToMigrate() || areThereAnyReadyTasks() || (atg && atg->isFinished());});
    updateWorkerCounters<false, true>(worker.getType(), -1);
}
