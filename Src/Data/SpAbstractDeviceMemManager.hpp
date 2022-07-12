#ifndef SPABSTRACTDEVICEMEMMANAGER_HPP
#define SPABSTRACTDEVICEMEMMANAGER_HPP

#if !(defined(SPECX_COMPILE_WITH_CUDA) || define(SPECX_COMPILE_WITH_HIP))
#error CUDA or HIP MUST BE ON
#endif

#include <list>

class SpAbstractDeviceMemManager {
public:
    virtual ~SpAbstractDeviceMemManager(){}

    virtual void incrDeviceDataUseCount(void* key) = 0;

    virtual void decrDeviceDataUseCount(void* key) = 0;

    virtual bool hasEnoughSpace(std::size_t inByteSize) = 0;

    virtual std::list<void*> candidatesToBeRemoved(const std::size_t inByteSize) = 0;

    virtual void* allocateWithKey(void* key, std::size_t inByteSize,
                          std::size_t alignment) = 0;

    virtual std::size_t freeGroup(void* key) = 0;

    virtual void memset(void* inPtrDev, const int val, const std::size_t inByteSize) = 0;

    virtual void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize) = 0;

    virtual void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize) = 0;

    virtual void copyDeviceToDevice(void* inPtrDeviceDest, const void* inPtrDeviceSrc, const int idSrc, const std::size_t inByteSize) = 0;
};

#endif
