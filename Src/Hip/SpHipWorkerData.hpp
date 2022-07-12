#ifndef SPHIPWORKERDATA_HPP
#define SPHIPWORKERDATA_HPP

#include <cassert>

#include <Config/SpConfig.hpp>

#include "SpHipUtils.hpp"

#ifndef SPECX_COMPILE_WITH_HIP
#error SPECX_COMPILE_WITH_HIP must be defined
#endif

struct SpHipWorkerData {
    int hipId = -1;
    hipStream_t stream;

    void init(int deviceId){
        hipId = deviceId;
    }

    void initByWorker(){
        assert(hipId != -1);
        SpHipUtils::UseDevice(hipId);
        HIP_ASSERT(hipStreamCreate(&stream));
    }

    void destroyByWorker(){
        HIP_ASSERT(hipStreamDestroy(stream));
    }

    void synchronize(){
        SpHipUtils::SynchronizeStream(stream);
    }
};


#endif // SPHIPWORKERDATA_HPP
