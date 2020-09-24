#include "SpAbstractTaskGraph.hpp"
#include "Compute/SpWorker.hpp"

void SpAbstractTaskGraph::finish() {
    auto workerForThread = SpWorker::getWorkerForThread();
    
    assert(workerForThread && "workerForThread is nullptr");
    
    workerForThread->doLoop(this);
}
