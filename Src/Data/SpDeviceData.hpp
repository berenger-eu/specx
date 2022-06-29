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
    virtual bool hasEnoughSpace(SpAbstractDeviceAllocator& allocator, void* rawHostPtr)  = 0;
    virtual std::list<void*> candidatesToBeRemoved(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) = 0;
    virtual SpDeviceData  allocate(SpAbstractDeviceAllocator& allocator, void* hostPtr) = 0;
    virtual void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, SpDeviceData devicePtr, void* hostPtr) = 0;
    virtual void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* hostPtr, SpDeviceData devicePtr) = 0;
    virtual void  copyFromDeviceToDevice(SpAbstractDeviceAllocator& allocator, SpDeviceData devicePtrDst, SpDeviceData devicePtrSrc, int srcId) = 0;
    virtual void  freeGroup(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) = 0;
};

//////////////////////////////////////////////////////////

template<class T> struct is_stdvector : public std::false_type {};

template<class T, class Alloc>
struct is_stdvector<std::vector<T, Alloc>> : public std::true_type {
    using _T = T;
};

//template<class K, class T, class Comp, class Alloc>
//struct is_container<std::map<K, T, Comp, Alloc>> : public std::true_type {};

//////////////////////////////////////////////////////////

template <class DataType>
class SpDeviceDataCopier : public SpAbstractDeviceDataCopier {
    template <typename T>
    using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copy_constructible<T>, std::is_trivially_copy_assignable<T>>;

public:
    bool hasEnoughSpace(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.hasEnoughSpace(sizeof(DataType));
        }
        else if constexpr(is_stdvector<DataType>::value){
            return allocator.hasEnoughSpace(sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
        }
        else {
            return false;// TODO
        }
    }

    std::list<void*> candidatesToBeRemoved(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return allocator.candidatesToBeRemoved(sizeof(DataType));
        }
        else if constexpr(is_stdvector<DataType>::value){
            return allocator.candidatesToBeRemoved(sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
        }
        else {
            return std::list<void*>();// TODO
        }
    }


    SpDeviceData allocate(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) override{
        SpDeviceData copy;
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            copy.ptr =  allocator.allocateWithKey(rawHostPtr, sizeof(DataType), alignof(DataType));
            copy.size = sizeof(DataType);
        }
        else if constexpr(is_stdvector<DataType>::value){
            copy.ptr =  allocator.allocateWithKey(rawHostPtr, sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size(), alignof(DataType));
            copy.size = sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size();
        }
        else {
//            copy.ptr =  SpDeviceDataAlloc(hostPtr);
//            copy.size = 0;
        }
        return copy;
    }
    void  copyFromHostToDevice(SpAbstractDeviceAllocator& allocator, SpDeviceData devicePtr, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtr.size == sizeof(DataType));
            return allocator.copyHostToDevice(devicePtr.ptr, hostPtr,  sizeof(DataType));
        }
        else if constexpr(is_stdvector<DataType>::value){
            assert(devicePtr.size == sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
            return allocator.copyHostToDevice(devicePtr.ptr, ((DataType*)rawHostPtr)->data(),
                                              devicePtr.size);
        }
        else {
            //SpDeviceDataCopyFromHostToDevice(devicePtr, hostPtr);
        }
    }
    void  copyFromDeviceToHost(SpAbstractDeviceAllocator& allocator, void* rawHostPtr, SpDeviceData devicePtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtr.size == sizeof(DataType));
            return allocator.copyDeviceToHost(hostPtr, devicePtr.ptr,  sizeof(DataType));
        }
        else if constexpr(is_stdvector<DataType>::value){
            assert(devicePtr.size == sizeof(typename is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
            return allocator.copyDeviceToHost(((DataType*)rawHostPtr)->data(), devicePtr.ptr,
                                              devicePtr.size);
        }
        else {
            //SpDeviceDataCopyFromDeviceToHost(hostPtr, devicePtr);
        }
    }
    void  copyFromDeviceToDevice(SpAbstractDeviceAllocator& allocator, SpDeviceData devicePtrDest, SpDeviceData devicePtrSrc, int srcId) override{
        if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtrDest.size == sizeof(DataType));
            assert(devicePtrDest.size == devicePtrSrc.size);
            return allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId, sizeof(DataType));
        }
        else if constexpr(is_stdvector<DataType>::value){
            assert(devicePtrDest.size == devicePtrSrc.size);
            return allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId,
                                              devicePtrDest.size);
        }
        else {
            //SpDeviceDataCopyFromDeviceToDevice(devicePtrDest, devicePtrSrc);
        }
    }
    void  freeGroup(SpAbstractDeviceAllocator& allocator, void* rawHostPtr) override{
        allocator.freeGroup(rawHostPtr);
    }
};


#endif
