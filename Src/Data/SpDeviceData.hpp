#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#include <cassert>
#include <type_traits>
#include <vector>
#include "SpAbstractDeviceMemManager.hpp"

struct SpDeviceData {
    void* ptr  = nullptr;
    void* viewPtr = nullptr;
    std::size_t size = 0;
};

//////////////////////////////////////////////////////////
namespace SpDeviceDataUtils{

template <typename, typename = std::void_t<>>
struct class_has_memmovNeededSize
: public std::false_type {};

template <typename Class>
struct class_has_memmovNeededSize<Class,
    std::void_t<decltype(std::declval<Class>().memmovNeededSize())>>
: public std::is_same<decltype(std::declval<Class>().memmovNeededSize()), std::size_t>
{};

///////////////////

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovHostToDevice
: public std::false_type {};

template <typename Class, typename CopierClass>
struct class_has_memmovHostToDevice<Class, CopierClass,
    std::void_t<decltype(std::declval<Class>().template memmovHostToDevice<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<std::size_t>()))>>
: public std::is_same<decltype(std::declval<Class>().template memmovHostToDevice<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<std::size_t>())), typename Class::DataDescriptor>
{};

///////////////////

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovDeviceToDevice
: public std::false_type {};

template <typename Class, typename CopierClass>
struct class_has_memmovDeviceToDevice<Class, CopierClass,
    std::void_t<decltype(std::declval<Class>().template memmovDeviceToDevice<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<void*>(), std::declval<int>(), std::declval<std::size_t>(), std::declval<const typename Class::DataDescriptor&>()))>>
: public std::is_same<decltype(std::declval<Class>().template memmovDeviceToDevice<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<void*>(), std::declval<int>(), std::declval<std::size_t>(), std::declval<const typename Class::DataDescriptor&>())), typename Class::DataDescriptor>
{};

///////////////////

template <typename, typename, typename = std::void_t<>>
struct class_has_memmovDeviceToHost
: public std::false_type {};

template <typename Class, typename CopierClass>
struct class_has_memmovDeviceToHost<Class, CopierClass,
    std::void_t<decltype(std::declval<Class>().template memmovDeviceToHost<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<std::size_t>(), std::declval<const typename Class::DataDescriptor&>()))>>
: public std::is_same<decltype(std::declval<Class>().template memmovDeviceToHost<CopierClass>(std::declval<CopierClass&>(), std::declval<void*>(), std::declval<std::size_t>(), std::declval<const typename Class::DataDescriptor&>())), void>
{};

///////////////////

template<class T> struct is_stdvector : public std::false_type {};

template<class T, class Alloc>
struct is_stdvector<std::vector<T, Alloc>> : public std::true_type {
    using _T = T;
};

template<class T, class Alloc>
struct is_stdvector<const std::vector<T, Alloc>> : public std::true_type {
    using _T = T;
};

///////////////////

//template<class K, class T, class Comp, class Alloc>
//struct is_container<std::map<K, T, Comp, Alloc>> : public std::true_type {};


///////////////////
template <typename T>
using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copyable<T>>;

///////////////////

template<class T> struct is_trivial_stdvector : public std::false_type {};

template<class T, class Alloc>
struct is_trivial_stdvector<std::vector<T, Alloc>> : public SpDeviceDataTrivialCopyTest<T> {
    using _T = T;
};

template<class T, class Alloc>
struct is_trivial_stdvector<const std::vector<T, Alloc>> : public SpDeviceDataTrivialCopyTest<T> {
    using _T = T;
};

///////////////////

// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf.
template <typename...>
using void_t = void;

// Primary template handles all types not supporting the operation.
template <typename, template <typename> class, typename = void_t<>>
struct detect : std::false_type {};

// Specialization recognizes/validates only types supporting the archetype.
template <typename T, template <typename> class Op>
struct detect<T, Op, void_t<Op<T>>> : std::true_type {};

template <typename T>
using setDataDescriptor_test = decltype(std::declval<T>().setDataDescriptor(nullptr));

template <typename T>
using class_has_setDataDescriptor = detect<T, setDataDescriptor_test>;

///////////////////
template <class AllocatorClass>
class SpAllocatorInterface{
    AllocatorClass& memManager;
    void* basedDevicePtr;
    const std::size_t memBlockSize;

public:
    SpAllocatorInterface(AllocatorClass& inMemManager, void* inBasedDevicePtr, const std::size_t inMemBlockSize)
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
    void  copyDeviceToHost(void* hostPtr, void* devicePtr, std::size_t size){
        assert(std::size_t(devicePtr) >= std::size_t(basedDevicePtr)
               && std::size_t(devicePtr) + size <= std::size_t(basedDevicePtr) + memBlockSize);
        memManager.copyDeviceToHost(hostPtr, devicePtr, size);
    }
    void  copyDeviceToDevice(void* destPtr, const void* srcPtr, const int srcDevId, std::size_t size){
        assert(std::size_t(destPtr) >= std::size_t(basedDevicePtr)
               && std::size_t(destPtr) + size <= std::size_t(basedDevicePtr) + memBlockSize);
        memManager.copyDeviceToDevice(destPtr, srcPtr, srcDevId, size);
    }
};

///////////////////

enum class DeviceMovableType{
    RAW_COPY,
    STDVEC,
    MEMMOV,
    ERROR
};

template <class RawDataType>
DeviceMovableType constexpr GetDeviceMovableType(){
    using DataType = std::remove_const_t<std::remove_reference_t<RawDataType>>;
    if constexpr(class_has_memmovNeededSize<DataType>::value
                && class_has_memmovHostToDevice<DataType, SpAllocatorInterface<SpAbstractDeviceMemManager>>::value
                && class_has_memmovDeviceToHost<DataType, SpAllocatorInterface<SpAbstractDeviceMemManager>>::value){
        return DeviceMovableType::MEMMOV;
    }
    else{
        static_assert(!class_has_memmovNeededSize<DataType>::value
                && !class_has_memmovHostToDevice<DataType, SpAllocatorInterface<SpAbstractDeviceMemManager>>::value
                && !class_has_memmovDeviceToHost<DataType, SpAllocatorInterface<SpAbstractDeviceMemManager>>::value,
                "Error, class has method of type MEMMOV but not all of them...");
        if constexpr(is_trivial_stdvector<DataType>::value){
            return DeviceMovableType::STDVEC;
        }
        else if constexpr(SpDeviceDataTrivialCopyTest<DataType>::value) {
            return DeviceMovableType::RAW_COPY;
        }
        else {
            return DeviceMovableType::ERROR;
        }
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
     const static SpDeviceDataUtils::DeviceMovableType MoveType = SpDeviceDataUtils::DeviceMovableType::RAW_COPY;

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
    using ObjType = typename SpDeviceDataUtils::is_trivial_stdvector<DataType>::_T;
    void *rawPtr;
    std::size_t rawSize;
public:
     const static SpDeviceDataUtils::DeviceMovableType MoveType = SpDeviceDataUtils::DeviceMovableType::STDVEC;

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
    using DataDescriptor = typename DataType::DataDescriptor;
    void *rawPtr;
    std::size_t rawSize;
    const DataDescriptor* deviceData;

public:
    const static SpDeviceDataUtils::DeviceMovableType MoveType = SpDeviceDataUtils::DeviceMovableType::MEMMOV;

     SpDeviceDataView()
         : rawPtr(nullptr), rawSize(0), deviceData(0){
     }

     void reset(void* ptr, std::size_t size){
        rawPtr = ptr;
        rawSize = size;
     }

   void setDataDescriptor(const void* viewPtr){
        deviceData = reinterpret_cast<const DataDescriptor*>(viewPtr);
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

    const DataDescriptor& data() const{
        return *deviceData;
    }
};


template <class DataType>
class SpDeviceDataView<DataType,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::ERROR>>{
    void *rawPtr;
    std::size_t rawSize;
public:
     const static SpDeviceDataUtils::DeviceMovableType MoveType = SpDeviceDataUtils::DeviceMovableType::ERROR;

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

template <class AllocatorClass>
class SpAbstractDeviceDataCopier {
public:
    virtual ~SpAbstractDeviceDataCopier(){};
    virtual bool hasEnoughSpace(AllocatorClass& allocator, void* key, void* rawHostPtr)  = 0;
    virtual std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* key,  void* rawHostPtr) = 0;
    virtual SpDeviceData  allocate(AllocatorClass& allocator, void* key, void* hostPtr) = 0;
    virtual void*  copyHostToDevice(AllocatorClass& allocator, void* key, SpDeviceData devicePtr, void* hostPtr) = 0;
    virtual void  copyDeviceToHost(AllocatorClass& allocator, void* key, void* hostPtr, SpDeviceData devicePtr) = 0;
    virtual void*  copyFromDeviceToDevice(AllocatorClass& allocator, void* key, void* rawHostPtr, SpDeviceData devicePtrDst, SpDeviceData devicePtrSrc, int srcId) = 0;
    virtual void  freeGroup(AllocatorClass& allocator, void* key, void* rawHostPtr) = 0;
};


template <class DataType, class AllocatorClass, typename = std::void_t<>>
class SpDeviceDataCopier;


template <class DataType, class AllocatorClass>
class SpDeviceDataCopier<DataType, AllocatorClass,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::RAW_COPY>>
        :  public SpAbstractDeviceDataCopier<AllocatorClass> {
public:
       using DataType_t = DataType;

    bool hasEnoughSpace(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(DataType));
        return allocator.hasEnoughSpace(neededSize);
    }

    std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(DataType));
        return allocator.candidatesToBeRemoved(neededSize);
    }

    SpDeviceData allocate(AllocatorClass& allocator, void* key, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(DataType));
        SpDeviceData copy;
        copy.size = neededSize;
        copy.ptr =  allocator.allocateWithKey(key, copy.size, alignof(DataType));
        return copy;
    }

    void*  copyHostToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtr, void* rawHostPtr) override {
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        assert(devicePtr.size == sizeof(DataType));
            allocator.copyHostToDevice(devicePtr.ptr, hostPtr,  sizeof(DataType));
                                                                                                                                          return nullptr;
    }

    void  copyDeviceToHost(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr, SpDeviceData devicePtr) override {
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        assert(devicePtr.size == sizeof(DataType));
            return allocator.copyDeviceToHost(hostPtr, devicePtr.ptr,  sizeof(DataType));
    }

    void*  copyFromDeviceToDevice(AllocatorClass& allocator, void* /*key*/, void* /*rawHostPtr*/, SpDeviceData devicePtrDest, SpDeviceData devicePtrSrc, int srcId) override {
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtrDest.size == sizeof(DataType));
        }
        assert(devicePtrDest.size == devicePtrSrc.size);
        allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId,
                                          devicePtrDest.size);
                                                                                                                                          return nullptr;
    }

    void  freeGroup(AllocatorClass& allocator, void* key, void* /*rawHostPtr*/) override {
        allocator.freeGroup(key);
    }
};


template <class DataType, class AllocatorClass>
class SpDeviceDataCopier<DataType, AllocatorClass,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::STDVEC>>
                 :  public SpAbstractDeviceDataCopier<AllocatorClass> {
public:
              using DataType_t = DataType;
    bool hasEnoughSpace(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * (reinterpret_cast<DataType*>(rawHostPtr))->size());
        return allocator.hasEnoughSpace(neededSize);
    }

    std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * (reinterpret_cast<DataType*>(rawHostPtr))->size());
        return allocator.candidatesToBeRemoved(neededSize);
    }

    SpDeviceData allocate(AllocatorClass& allocator, void* key, void* rawHostPtr) override {
        std::size_t neededSize = 0;
        neededSize = (sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * (reinterpret_cast<DataType*>(rawHostPtr))->size());
        SpDeviceData copy;
        copy.size = neededSize;
        copy.ptr =  allocator.allocateWithKey(key, copy.size, alignof(DataType));
        return copy;
    }

    void*  copyHostToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtr, void* rawHostPtr) override {
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        assert(devicePtr.size == sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * (reinterpret_cast<DataType*>(rawHostPtr))->size());
            allocator.copyHostToDevice(devicePtr.ptr, (reinterpret_cast<DataType*>(rawHostPtr))->data(),
                                              devicePtr.size);
                                                                                                                                        return nullptr;
    }

    void  copyDeviceToHost(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr, SpDeviceData devicePtr) override {
        DataType* hostPtr = static_cast<DataType*>(rawHostPtr);
        assert(devicePtr.size == sizeof(typename SpDeviceDataUtils::is_stdvector<DataType>::_T) * (reinterpret_cast<DataType*>(rawHostPtr))->size());
        allocator.copyDeviceToHost((reinterpret_cast<DataType*>(rawHostPtr))->data(), devicePtr.ptr,
                                              devicePtr.size);
    }

    void*  copyFromDeviceToDevice(AllocatorClass& allocator, void* /*key*/, void* /*rawHostPtr*/, SpDeviceData devicePtrDest, SpDeviceData devicePtrSrc, int srcId) override {
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtrDest.size == sizeof(DataType));
        }
        assert(devicePtrDest.size == devicePtrSrc.size);
        allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId,
                                          devicePtrDest.size);
                                                                                                                                        return nullptr;
    }

    void  freeGroup(AllocatorClass& allocator, void* key, void* /*rawHostPtr*/) override {
        allocator.freeGroup(key);
    }
};

template <class DataType, class AllocatorClass>
class SpDeviceDataCopier<DataType, AllocatorClass,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::MEMMOV>>
                      :  public SpAbstractDeviceDataCopier<AllocatorClass> {
public:
    using DataType_t = DataType;

    bool hasEnoughSpace(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
            neededSize = (reinterpret_cast<DataType*>(rawHostPtr))->memmovNeededSize();
        return allocator.hasEnoughSpace(neededSize);
    }
    std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        std::size_t neededSize = 0;
            neededSize = (reinterpret_cast<DataType*>(rawHostPtr))->memmovNeededSize();
        return allocator.candidatesToBeRemoved(neededSize);
    }
    SpDeviceData allocate(AllocatorClass& allocator, void* key, void* rawHostPtr) override {
        std::size_t neededSize = (reinterpret_cast<DataType*>(rawHostPtr))->memmovNeededSize();
        SpDeviceData copy;
        copy.size = neededSize;
        copy.ptr =  allocator.allocateWithKey(key, copy.size, alignof(DataType));
        return copy;
    }
    void*  copyHostToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtr, void* rawHostPtr) override {
        DataType* hostPtr = reinterpret_cast<DataType*>(rawHostPtr);
        SpDeviceDataUtils::SpAllocatorInterface<AllocatorClass> interface(allocator, devicePtr.ptr, devicePtr.size);
        auto view = hostPtr->memmovHostToDevice(interface, devicePtr.ptr, devicePtr.size);
        return new decltype(view)(view);
    }
    void  copyDeviceToHost(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr, SpDeviceData devicePtr) override {
        DataType* hostPtr = reinterpret_cast<DataType*>(rawHostPtr);
            SpDeviceDataUtils::SpAllocatorInterface<AllocatorClass> interface(allocator, devicePtr.ptr, devicePtr.size);
        const typename DataType::DataDescriptor& view = *(reinterpret_cast<typename DataType::DataDescriptor*>(devicePtr.viewPtr));
        hostPtr->memmovDeviceToHost(interface, devicePtr.ptr, devicePtr.size, view);
    }
    void*  copyFromDeviceToDevice(AllocatorClass& allocator, void* /*key*/, [[maybe_unsused]] void* rawHostPtr,
                                  SpDeviceData devicePtrDest, SpDeviceData devicePtrSrc, int srcId) override {
        if constexpr(SpDeviceDataUtils::SpDeviceDataTrivialCopyTest<DataType>::value) {
            assert(devicePtrDest.size == sizeof(DataType));
        }
        assert(devicePtrDest.size == devicePtrSrc.size);
        const typename DataType::DataDescriptor& otherView = *(reinterpret_cast<typename DataType::DataDescriptor*>(devicePtrSrc.viewPtr));
        if constexpr(SpDeviceDataUtils::class_has_memmovDeviceToDevice<DataType, SpDeviceDataUtils::SpAllocatorInterface<SpAbstractDeviceMemManager>>::value){
            DataType* hostPtr = reinterpret_cast<DataType*>(rawHostPtr);
            const typename DataType::DataDescriptor& srcView = *(reinterpret_cast<typename DataType::DataDescriptor*>(devicePtrSrc.viewPtr));
            SpDeviceDataUtils::SpAllocatorInterface<AllocatorClass> interface(allocator, devicePtrDest.ptr, devicePtrDest.size);
            auto view = hostPtr->memmovDeviceToDevice(interface, devicePtrDest.ptr, devicePtrSrc.ptr, srcId, devicePtrDest.size, srcView);
            return new decltype(view)(view);
        }
        else{
            allocator.copyDeviceToDevice(devicePtrDest.ptr, devicePtrSrc.ptr, srcId,
                                              devicePtrDest.size);
            return new std::remove_const_t<std::remove_reference_t<decltype(otherView)>>(otherView);
        }
    }
    void  freeGroup(AllocatorClass& allocator, void* key, void* /*rawHostPtr*/) override {
        allocator.freeGroup(key);
    }
};

template <class DataType, class AllocatorClass>
class SpDeviceDataCopier<DataType, AllocatorClass,
        typename std::enable_if_t<SpDeviceDataUtils::GetDeviceMovableType<DataType>() == SpDeviceDataUtils::DeviceMovableType::ERROR>>
                       :  public SpAbstractDeviceDataCopier<AllocatorClass> {
public:
    using DataType_t = DataType;

    bool hasEnoughSpace(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        assert(0);
        return false;
    }
    std::list<void*> candidatesToBeRemoved(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr) override {
        assert(0);
        return std::list<void*>();
    }
    SpDeviceData allocate(AllocatorClass& allocator, void* key, void* rawHostPtr) override {
        assert(0);
        return SpDeviceData();
    }
    void*  copyHostToDevice(AllocatorClass& allocator, void* /*key*/, SpDeviceData devicePtr, void* rawHostPtr) override {
        assert(0);
                                                                                                                                       return nullptr;
    }
    void  copyDeviceToHost(AllocatorClass& allocator, void* /*key*/, void* rawHostPtr, SpDeviceData devicePtr) override {
        assert(0);
    }
    void*  copyFromDeviceToDevice(AllocatorClass& allocator, void* /*key*/, void* /*rawHostPtr*/, SpDeviceData /*devicePtrDest*/,
                                 SpDeviceData /*devicePtrSrc*/, int /*srcId*/) override {
        assert(0);
        return nullptr;
    }
    void  freeGroup(AllocatorClass& allocator, void* /*key*/, void* /*rawHostPtr*/) override {
        assert(0);
    }
};



#endif
