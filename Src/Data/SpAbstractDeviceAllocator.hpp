#ifndef SPABSTRACTDEVICEALLOCATOR_HPP
#define SPABSTRACTDEVICEALLOCATOR_HPP

class SpDataHandle;

class SpAbstractDeviceAllocator {
public:
    virtual ~SpAbstractDeviceAllocator(){}

    virtual void* allocateWithKey(const SpDataHandle* key, std::size_t inByteSize,
                          std::size_t alignment) = 0;

    virtual void freeGroup(const SpDataHandle* key) = 0;

    virtual void memset(void* inPtrDev, const int val, const std::size_t inByteSize) = 0;

    virtual void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize) = 0;

    virtual void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize) = 0;
};

#endif
