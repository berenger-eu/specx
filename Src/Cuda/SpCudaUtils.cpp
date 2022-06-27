#include "SpCudaUtils.hpp"
#include "Compute/SpWorker.hpp"

std::vector<bool> SpCudaUtils::ConnectedDevices = SpCudaUtils::ConnectDevices();


cudaStream_t& SpCudaUtils::GetCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsGpu());
    return SpWorker::getWorkerForThread()->getGpuData().stream;
}

bool SpCudaUtils::CurrentWorkerIsGpu(){
    assert(SpWorker::getWorkerForThread());
    return SpWorker::getWorkerForThread()->getType() == SpWorker::SpWorkerType::GPU_WORKER;
}

int SpCudaUtils::CurrentGpuId(){
    assert(SpWorker::getWorkerForThread());
    return SpWorker::getWorkerForThread()->getGpuData().gpuId;
}

void SpCudaUtils::SyncCurrentStream(){
    assert(SpWorker::getWorkerForThread());
    assert(CurrentWorkerIsGpu());
    return SpWorker::getWorkerForThread()->getGpuData().synchronize();
}
