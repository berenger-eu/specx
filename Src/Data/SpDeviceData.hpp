#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#ifndef SPECX_COMPILE_WITH_CUDA
#error CUDE MUST BE ON
#endif

#include <type_traits>
#include "SpAbstractDeviceMemManager.hpp"

struct SpDeviceData {
    void* ptr = nullptr;
    std::size_t size;
};


template <class AllocatorClass>
class SpAbstractDeviceDataCopier {
public:
    virtual ~SpAbstractDeviceDataCopier(){};
    virtual bool hasEnoughSpace(AllocatorClass& allocator, void* key, void* rawHostPtr)  = 0;
    virtual std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* key,  void* rawHostPtr) = 0;
    virtual SpDeviceData  allocate(AllocatorClass& allocator, void* key, void* hostPtr) = 0;
    virtual void  copyFromHostToDevice(AllocatorClass& allocator, void* key, SpDeviceData devicePtr, void* hostPtr) = 0;
    virtual void  copyFromDeviceToHost(AllocatorClass& allocator, void* key, void* hostPtr, SpDeviceData devicePtr) = 0;
    virtual void  copyFromDeviceToDevice(AllocatorClass& allocator, void* key, SpDeviceData devicePtrDst, SpDeviceData devicePtrSrc, int srcId) = 0;
    virtual void  freeGroup(AllocatorClass& allocator, void* key, void* rawHostPtr) = 0;
};


//////////////////////////////////////////////////////////
namespace SpDeviceDataUtils{

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovNeededSize
: public std::false_type {};

template <typename Class>
struct class_has_memmovNeededSize<Class,
    std::void_t<decltype(std::declval<Class>().memmovNeededSize(std::declval<void>()))>>
: public std::is_same<decltype(std::declval<Class>().memmovNeededSize(std::declval<void>())), std::size_t>
{};

///////////////////

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovHostToDevice
: public std::false_type {};

template <typename Class, typename CopierClass>
struct class_has_memmovHostToDevice<Class, CopierClass,
    std::void_t<decltype(std::declval<Class>().memmovHostToDevice(std::declval<CopierClass, void*, std::size_t>()))>>
: public std::is_same<decltype(std::declval<Class>().memmovHostToDevice(std::declval<CopierClass, void*, std::size_t>())), void>
{};

///////////////////

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovDeviceToHost
: public std::false_type {};

template <typename Class, typename CopierClass>
struct class_has_memmovDeviceToHost<Class, CopierClass,
    std::void_t<decltype(std::declval<Class>().memmovDeviceToHost(std::declval<CopierClass, void*, std::size_t>()))>>
: public std::is_same<decltype(std::declval<Class>().memmovDeviceToHost(std::declval<CopierClass, void*, std::size_t>())), void>
{};

///////////////////

template<class T> struct is_stdvector : public std::false_type {};

template<class T, class Alloc>
struct is_stdvector<std::vector<T, Alloc>> : public std::true_type {
    using _T = T;
};

///////////////////

//template<class K, class T, class Comp, class Alloc>
//struct is_container<std::map<K, T, Comp, Alloc>> : public std::true_type {};


///////////////////
template <typename T>
using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copyable<T>>;


///////////////////
template <class AllocatorClass>
class SpDeviceMemmov{
    AllocatorClass& memManager;
    void* basedDevicePtr;
    const std::size_t memBlockSize;

public:
    SpDeviceMemmov(AllocatorClass& inMemManager, void* inBasedDevicePtr, const std::size_t inMemBlockSize)
        : memManager(inMemManager), basedDevicePtr(inBasedDevicePtr), memBlockSize(inMemBlockSize){}

    void* ptr() const{
        return basedDevicePtr;
    }

    std::size_t blockSize() const{
        return memBlockSize;
    }

    void  copyHostToDevice(void* devicePtr, void* hostPtr, std::size_t size){
        assert(std::size_t(devicePtr) >= std::size_t(basedDevicePtr)
               && std::size_t(devicePtr) + size <= std::size_t(basedDevicePtr) + memBlockSize);
        memManager.copyHostToDevice(devicePtr, hostPtr, size);
    }
    void  copyFromDeviceToHost(void* hostPtr, void* devicePtr, std::size_t size){
        assert(std::size_t(devicePtr) >= std::size_t(basedDevicePtr)
               && std::size_t(devicePtr) + size <= std::size_t(basedDevicePtr) + memBlockSize);
        memManager.DeviceToHost(hostPtr, devicePtr, size);
    }
};

///////////////////

enum class DeviceMovableType{
    RAW_COPY,
    STDVEC,
    MEMMOV,
    ERROR
};

template <class DataType>
DeviceMovableType constexpr GetDeviceMovableType(){
    if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
        return DeviceMovableType::RAW_COPY;
    }
    else if constexpr(is_stdvector<DataType>::value){
        return DeviceMovableType::STDVEC;
    }
    else if constexpr(class_has_memmovNeededSize<DataType, SpDeviceMemmov<SpAbstractDeviceMemManager>>::value
            && class_has_memmovHostToDevice<DataType, SpDeviceMemmov<SpAbstractDeviceMemManager>>::value
            && class_has_memmovDeviceToHost<DataType, SpDeviceMemmov<SpAbstractDeviceMemManager>>::value){
        return DeviceMovableType::MEMMOV;
    }
    else{
        return DeviceMovableType::ERROR;
    }
}

template <class DataType>
bool constexpr IsDeviceMovable(){
    return GetDeviceMovableType<DataType>() != DeviceMovableType::ERROR;
}

}
////////////////////////////////////////////////////////////////

template <class DataType, typename = std::void_t<>>
class SpDeviceDataView;


template <class DataType>
class SpDeviceDataView<DataType,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::RAW_COPY>>{

    using CleanDataType = std::remove_reference_t<DataType>;

    void *rawPtr;
    std::size_t rawSize;
public:
     SpDeviceDataView()
         : rawPtr(nullptr), rawSize(0){
     }

     void reset(void* ptr, std::size_t size){
        rawPtr = ptr;
        rawSize = size;
     }

    void* getRawPtr(){
        return rawPtr;
    }

    const void* getRawPtr() const{
        return rawPtr;
    }

    std::size_t getRawSize(){
        return rawSize;
    }

    CleanDataType* objPtr(){
        return reinterpret_cast<CleanDataType*>(rawPtr);
    }

    const CleanDataType* objPtr() const {
        return reinterpret_cast<const CleanDataType*>(rawPtr);
    }
};


template <class DataType>
class SpDeviceDataView<DataType,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::STDVEC>>{
    using ObjType = typename SpDeviceDataUtils::is_stdvector<DataType>::_T;
    void *rawPtr;
    std::size_t rawSize;
public:
     SpDeviceDataView()
         : rawPtr(nullptr), rawSize(0){
     }

     void reset(void* ptr, std::size_t size){
        rawPtr = ptr;
        rawSize = size;
     }

    void* getRawPtr(){
        assert(0);
        return rawPtr;
    }

    const void* getRawPtr() const{
        assert(0);
        return rawPtr;
    }

    std::size_t getRawSize(){
        assert(0);
        return rawSize;
    }

    ObjType* array(){
        return reinterpret_cast<ObjType*>(rawPtr);
    }

    const ObjType* array() const {
        return reinterpret_cast<const ObjType*>(rawPtr);
    }

    long int nbElements() const{
        return rawSize/sizeof(ObjType);
    }
};



template <class DataType>
class SpDeviceDataView<DataType,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::MEMMOV>>{
    using DeviceDataType = typename DataType::DeviceDataType;
    void *rawPtr;
    std::size_t rawSize;
    DeviceDataType deviceData;
public:
     SpDeviceDataView()
         : rawPtr(nullptr), rawSize(0){
     }

     void reset(void* ptr, std::size_t size){
        rawPtr = ptr;
        rawSize = size;
        deviceData = DeviceDataType(ptr, size);
     }

    void* getRawPtr(){
        assert(0);
        return rawPtr;
    }

    const void* getRawPtr() const{
        assert(0);
        return rawPtr;
    }

    std::size_t getRawSize(){
        assert(0);
        return rawSize;
    }

    DeviceDataType& data(){
        return deviceData;
    }

    const DeviceDataType& data() const {
        return deviceData;
    }
};


template <class DataType>
class SpDeviceDataView<DataType,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::ERROR>>{
    void *rawPtr;
    std::size_t rawSize;
public:
     SpDeviceDataView()
         : rawPtr(nullptr), rawSize(0){
     }

     void reset(void* ptr, std::size_t size){
        rawPtr = ptr;
        rawSize = size;
     }

    void* getRawPtr(){
        assert(0);
        return rawPtr;
    }

    const void* getRawPtr() const{
        assert(0);
        return rawPtr;
    }

    std::size_t getRawSize(){
        assert(0);
        return rawSize;
    }
};



template <typename OrignalTuple, std::size_t... Is>
auto DeviceViewTyple_Core(std::index_sequence<Is...>) {
    return std::tuple<SpDeviceDataView<std::remove_reference_t<decltype(std::declval<std::tuple_element_t<Is, std::decay_t<OrignalTuple>>>().getView())>>...>();
}

template <typename OrignalTuple>
auto DeviceViewTyple_Core() {
  return DeviceViewTyple_Core<OrignalTuple>(std::make_index_sequence<std::tuple_size<std::decay_t<OrignalTuple>>::value>());
}

template <class OrignalTuple>
using DeviceViewTyple = decltype (DeviceViewTyple_Core<OrignalTuple>());

//////////////////////////////////////////////////////////

template <class DataType, class AllocatorClass>
class SpDeviceDataCopier : public SpAbstractDeviceDataCopier<AllocatorClass> {
public:
    bool hasEnoughSpace(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override{
        std::size_t neededSize = 0;
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            neededSize = (sizeof(DataType));
        }
        else if constexpr(SpDeviceDataUtils::is_stdvector<DataType>::value){
            neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
        }
        else if constexpr(SpDeviceDataUtils::class_has_memmovNeededSize<DataType, SpDeviceDataUtils::SpDeviceMemmov<AllocatorClass>>::value){
            neededSize = ((DataType*)rawHostPtr)->memmovNeededSize();
        }
        else{
            assert(0);
        }
        return allocator.hasEnoughSpace(neededSize);
    }

    std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override{
        std::size_t neededSize = 0;
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            neededSize = (sizeof(DataType));
        }
        else if constexpr(SpDeviceDataUtils::is_stdvector<DataType>::value){
            neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
        }
        else if constexpr(SpDeviceDataUtils::class_has_memmovNeededSize<DataType, SpDeviceDataUtils::SpDeviceMemmov<AllocatorClass>>::value){
            neededSize = ((DataType*)rawHostPtr)->memmovNeededSize();
        }
        else{
            assert(0);
        }
        return allocator.candidatesToBeRemoved(neededSize);
    }


    SpDeviceData allocate(AllocatorClass& allocator, void* key, void* rawHostPtr) override{
        std::size_t neededSize = 0;
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            neededSize = (sizeof(DataType));
        }
        else if constexpr(SpDeviceDataUtils::is_stdvector<DataType>::value){
            neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
        }
        else if constexpr(SpDeviceDataUtils::class_has_memmovNeededSize<DataType, SpDeviceDataUtils::SpDeviceMemmov<AllocatorClass>>::value){
            neededSize = ((DataType*)rawHostPtr)->memmovNeededSize();
        }
        else{
            assert(0);
        }
        SpDeviceData copy;
        copy.size = neededSize;
        copy.ptr =  allocator.allocateWithKey(key, copy.size, alignof(DataType));
        return copy;
    }
    void  copyFromHostToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtr, void* rawHostPtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtr.size == sizeof(DataType));
            allocator.copyHostToDevice(devicePtr.ptr, hostPtr,  sizeof(DataType));
        }
        else if constexpr(SpDeviceDataUtils::is_stdvector<DataType>::value){
            assert(devicePtr.size == sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
            allocator.copyHostToDevice(devicePtr.ptr, ((DataType*)rawHostPtr)->data(),
                                              devicePtr.size);
        }
        else if constexpr(SpDeviceDataUtils::class_has_memmovHostToDevice<DataType, SpDeviceDataUtils::SpDeviceMemmov<AllocatorClass>>::value){
            ((DataType*)rawHostPtr)->memmovHostToDevice(devicePtr.ptr, devicePtr.size);
        }
        else {
            assert(0);
        }
    }
    void  copyFromDeviceToHost(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr, SpDeviceData devicePtr) override{
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtr.size == sizeof(DataType));
            return allocator.copyDeviceToHost(hostPtr, devicePtr.ptr,  sizeof(DataType));
        }
        else if constexpr(SpDeviceDataUtils::is_stdvector<DataType>::value){
            assert(devicePtr.size == sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * ((DataType*)rawHostPtr)->size());
            return allocator.copyDeviceToHost(((DataType*)rawHostPtr)->data(), devicePtr.ptr,
                                              devicePtr.size);
        }
        else if constexpr(SpDeviceDataUtils::class_has_memmovDeviceToHost<DataType, SpDeviceDataUtils::SpDeviceMemmov<AllocatorClass>>::value){
            ((DataType*)rawHostPtr)->memmovDeviceToHost(devicePtr.ptr, devicePtr.size);
        }
        else {
            assert(0);
        }
    }
    void  copyFromDeviceToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtrDest, SpDeviceData devicePtrSrc, int srcId) override{
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtrDest.size == sizeof(DataType));
        }
        assert(devicePtrDest.size == devicePtrSrc.size);
        allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId,
                                          devicePtrDest.size);
    }
    void  freeGroup(AllocatorClass& allocator, void* key, void* /*rawHostPtr*/) override{
        allocator.freeGroup(key);
    }
};


#endif
