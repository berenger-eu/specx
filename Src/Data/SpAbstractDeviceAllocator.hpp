#ifndef SPABSTRACTDEVICEALLOCATOR_HPP
#define SPABSTRACTDEVICEALLOCATOR_HPP

#ifndef SPETABARU_COMPILE_WITH_CUDA
#error CUDE MUST BE ON
#endif

class SpAbstractDeviceAllocator {
public:
    virtual ~SpAbstractDeviceAllocator(){}

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

    virtual void copyDeviceToDevice(void* inPtrDeviceDest, const void* inPtrDeviceSrc, const std::size_t inByteSize) = 0;
};

#endif
