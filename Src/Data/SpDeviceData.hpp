#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#include <type_traits>
#include "Utils/SpDeviceAllocator.hpp"
#include "Utils/SpGpuUnusedDataStore.hpp"

struct SpDeviceData {
    void* ptr = nullptr;
    long int useCount = 0;
};

class SpAbstractDeviceDataCopier {
private:
    template <typename T>
    using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copy_constructible<T>, std::is_trivially_copy_assignable<T>>;

public:
    virtual ~SpAbstractDeviceDataCopier(){};
    virtual void  allocate(void* hostPtr) = 0;
    virtual void  copyFromHostToDevice(void* devicePtr, void* hostPtr) = 0;
    virtual void  copyFromDeviceToHost(void* hostPtr, void* devicePtr) = 0;
    virtual void  free(void* devicePtr) = 0;
};

template <class DataType>
class SpDeviceDataCopier : public SpAbstractDeviceDataCopier {
public:
    SpDeviceData allocate(void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        SpDeviceData copy;
        copy.ptr = SpDeviceDataAllocInternal(static_cast<DataType*>(hostPtr));
        if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
            copy.ptr =  SpDeviceAllocator::allocateOnDevice(sizeof(DataType), alignof(DataType));
        } else {
            copy.ptr =  SpDeviceDataAlloc(hostPtr);
        }
        return copy;
    }
    void  copyFromHostToDevice(void* devicePtr, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return SpDeviceAllocator::copyHostToDevice(devicePtr, hostPtr,  sizeof(DataType));
        } else {
            SpDeviceDataCopyFromHostToDevice(devicePtr, hostPtr);
        }
    }
    void  copyFromDeviceToHost(void* rawHostPtr, void* devicePtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return SpDeviceAllocator::copyDeviceToHost(hostPtr, devicePtr,  sizeof(DataType));
        } else {
            SpDeviceDataCopyFromDeviceToHost(hostPtr, devicePtr);
        }
    }
    void  free(void* devicePtr) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return SpDeviceAllocator::freeFromDevice(devicePtr);
        } else {
            SpDeviceDataFree<DataType>(devicePtr);
        }
    }
};


#endif
