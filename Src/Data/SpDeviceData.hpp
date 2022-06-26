#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#ifndef SPETABARU_COMPILE_WITH_CUDA
#error CUDE MUST BE ON
#endif

#include <type_traits>
#include "SpAbstractDeviceAllocator.hpp"

struct SpDeviceData {
    void* ptr = nullptr;
    std::size_t size;
};

class SpAbstractDeviceDataCopier {
public:
    virtual ~SpAbstractDeviceDataCopier(){};
    virtual bool hasEnoughSpace(SpAbstractDeviceAllocator& allocator, void* key)  = 0;
    virtual std::list<void*> candidatesToBeRemoved(SpAbstractDeviceAllocator& allocator, void* key) = 0;
    virtual SpDeviceData  allocate(SpAbstractDeviceAllocator& allocator, void* hostPtr) = 0;
    virtual void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtr, void* hostPtr) = 0;
    virtual void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* hostPtr, void* devicePtr) = 0;
    virtual void  copyFromDeviceToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtrDst, void* devicePtrSrc) = 0;
    virtual void  freeGroup(SpAbstractDeviceAllocator& allocator, void* key) = 0;
};

template <class DataType>
class SpDeviceDataCopier : public SpAbstractDeviceDataCopier {
    template <typename T>
    using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copy_constructible<T>, std::is_trivially_copy_assignable<T>>;

public:
    bool hasEnoughSpace(SpAbstractDeviceAllocator& allocator, void* /*key*/) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.hasEnoughSpace(sizeof(DataType));
        }
        else {
            return false;// TODO
        }
    }

    std::list<void*> candidatesToBeRemoved(SpAbstractDeviceAllocator& allocator, void* /*key*/) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.candidatesToBeRemoved(sizeof(DataType));
        }
        else {
            return std::list<void*>();// TODO
        }
    }


    SpDeviceData allocate(SpAbstractDeviceAllocator& allocator, void* key) override{
        SpDeviceData copy;
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            copy.ptr =  allocator.allocateWithKey(key, sizeof(DataType), alignof(DataType));
            copy.size = sizeof(DataType);
        }
        else {
//            copy.ptr =  SpDeviceDataAlloc(hostPtr);
//            copy.size = 0;
        }
        return copy;
    }
    void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtr, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyHostToDevice(devicePtr, hostPtr,  sizeof(DataType));
        }
        else {
            //SpDeviceDataCopyFromHostToDevice(devicePtr, hostPtr);
        }
    }
    void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* rawHostPtr, void* devicePtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyDeviceToHost(hostPtr, devicePtr,  sizeof(DataType));
        }
        else {
            //SpDeviceDataCopyFromDeviceToHost(hostPtr, devicePtr);
        }
    }
    void  copyFromDeviceToDevice(SpAbstractDeviceAllocator& allocator, void* devicePtrDest, void* devicePtrSrc) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.copyDeviceToDevice(devicePtrDest, devicePtrSrc,  sizeof(DataType));
        }
        else {
            //SpDeviceDataCopyFromDeviceToDevice(devicePtrDest, devicePtrSrc);
        }
    }
    void  freeGroup(SpAbstractDeviceAllocator& allocator, void* key) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            allocator.freeGroup(key);
        }
        else {
            //SpDeviceDataFree<DataType>(devicePtr);
        }
    }
};


#endif
