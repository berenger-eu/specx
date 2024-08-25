#include "SpHipUtils.hpp"
#include "Compute/SpWorker.hpp"

std::vector<bool> SpHipUtils::ConnectedDevices = SpHipUtils::ConnectDevices();


hipStream_t& SpHipUtils::GetCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsHip());
    return SpWorker::getWorkerForThread()->getHipData().stream;
}

bool SpHipUtils::CurrentWorkerIsHip(){
    return SpWorker::getWorkerForThread() && SpWorker::getWorkerForThread()->getType() == SpWorkerTypes::Type::HIP_WORKER;
}

int SpHipUtils::CurrentHipId(){
    assert(SpWorker::getWorkerForThread());
    return SpWorker::getWorkerForThread()->getHipData().hipId;
}

void SpHipUtils::SyncCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsHip());
    return SpWorker::getWorkerForThread()->getHipData().synchronize();
}
