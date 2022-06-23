#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#include <type_traits>
#include "SpAbstractDeviceAllocator.hpp"

struct SpDeviceData {
    void* ptr = nullptr;
    long int useCount = 0;
};

class SpAbstractDeviceDataCopier {
public:
    virtual ~SpAbstractDeviceDataCopier(){};
    virtual void  allocate(SpAbstractDeviceAllocator& allocator, void* hostPtr) = 0;
    virtual void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtr, void* hostPtr) = 0;
    virtual void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* hostPtr, void* devicePtr) = 0;
    virtual void  free(SpAbstractDeviceAllocator& allocator, void* devicePtr) = 0;
};

template <class DataType>
class SpDeviceDataCopier : public SpAbstractDeviceDataCopier {
    template <typename T>
    using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copy_constructible<T>, std::is_trivially_copy_assignable<T>>;

public:
    SpDeviceData allocate(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        SpDeviceData copy;
        copy.ptr = SpDeviceDataAllocInternal(static_cast<DataType*>(hostPtr));
        if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
            copy.ptr =  allocator.allocateOnDevice(sizeof(DataType), alignof(DataType));
        } else {
            copy.ptr =  SpDeviceDataAlloc(hostPtr);
        }
        return copy;
    }
    void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtr, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyHostToDevice(devicePtr, hostPtr,  sizeof(DataType));
        } else {
            SpDeviceDataCopyFromHostToDevice(devicePtr, hostPtr);
        }
    }
    void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* rawHostPtr, void* devicePtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyDeviceToHost(hostPtr, devicePtr,  sizeof(DataType));
        } else {
            SpDeviceDataCopyFromDeviceToHost(hostPtr, devicePtr);
        }
    }
    void  copyFromDeviceToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtrDest, void* devicePtrSrc) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyDeviceToDevice(devicePtrDest, devicePtrSrc,  sizeof(DataType));
        } else {
            SpDeviceDataCopyFromDeviceToDevice(devicePtrDest, devicePtrSrc);
        }
    }
    void  free(SpAbstractDeviceAllocator& allocator, void* devicePtr) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.freeFromDevice(devicePtr);
        } else {
            SpDeviceDataFree<DataType>(devicePtr);
        }
    }
};


#endif
