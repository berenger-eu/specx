#include "SpCudaUtils.hpp"
#include "Compute/SpWorker.hpp"

std::vector<bool> SpCudaUtils::ConnectedDevices = SpCudaUtils::ConnectDevices();


cudaStream_t& SpCudaUtils::GetCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsCuda());
    return SpWorker::getWorkerForThread()->getCudaData().stream;
}

bool SpCudaUtils::CurrentWorkerIsCuda(){
    assert(SpWorker::getWorkerForThread());
    return SpWorker::getWorkerForThread()->getType() == SpWorkerTypes::Type::CUDA_WORKER;
}

int SpCudaUtils::CurrentCudaId(){
    assert(SpWorker::getWorkerForThread());
    return SpWorker::getWorkerForThread()->getCudaData().cudaId;
}

void SpCudaUtils::SyncCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsCuda());
    return SpWorker::getWorkerForThread()->getCudaData().synchronize();
}
