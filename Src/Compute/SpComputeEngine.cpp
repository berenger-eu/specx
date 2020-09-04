#include <optional>
#include <mutex>

#include "SpComputeEngine.hpp"
#include "TaskGraph/SpAbstractTaskGraph.hpp"
#include "Tasks/SpAbstractTask.hpp"
#include "SpWorker.hpp"

void SpComputeEngine::addGraph(SpAbstractTaskGraph* tg) {
    if(tg) {
        tg->setComputeEngine(this);
        taskGraphs.push_back(tg);
    }
}

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
