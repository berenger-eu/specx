#ifndef SPCUDAWORKERDATA_HPP
#define SPCUDAWORKERDATA_HPP

#include <cassert>

#include <Config/SpConfig.hpp>
#include <Utils/SpUtils.hpp>

#include "SpCudaUtils.hpp"

#ifndef SPECX_COMPILE_WITH_CUDA
#error SPECX_COMPILE_WITH_CUDA must be defined
#endif

struct SpCudaWorkerData {
    int cudaId = -1;
    cudaStream_t stream = nullptr;

    void init(int deviceId){
        cudaId = deviceId;
    }

    void initByWorker(){
        assert(cudaId != -1);
        SpCudaUtils::UseDevice(cudaId);
        CUDA_ASSERT(cudaStreamCreate(&stream));
        SpUtils::SetDeviceId(cudaId);
    }

    void destroyByWorker(){
        CUDA_ASSERT(cudaStreamDestroy(stream));
    }

    void synchronize(){
        SpCudaUtils::SynchronizeStream(stream);
    }
};


#endif // SPCUDAWORKERDATA_HPP
