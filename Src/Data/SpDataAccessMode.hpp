///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDATAACCESSMODE_HPP
#define SPDATAACCESSMODE_HPP

#include <type_traits>
#include <array>
#include <utility>

#include "Config/SpConfig.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpDebug.hpp"
#include "Data/SpDataDuplicator.hpp"
#include "Utils/SpArrayAccessor.hpp"
#include "Utils/small_vector.hpp"
#include "Utils/SpUtils.hpp"

////////////////////////////////////////////////////////
/// All possible data access modes
////////////////////////////////////////////////////////

enum class SpDataAccessMode{
    READ=0,
    WRITE,
    PARALLEL_WRITE,
    COMMUTATIVE_WRITE,
    POTENTIAL_WRITE
};

using SpDataAccessModeField = unsigned char;

////////////////////////////////////////////////////////
/// Convert to string
////////////////////////////////////////////////////////

inline std::string SpModeToStr(const SpDataAccessMode inMode){
    switch(inMode){
    case SpDataAccessMode::READ: return "READ";
    case SpDataAccessMode::WRITE: return "WRITE";
    case SpDataAccessMode::PARALLEL_WRITE: return "PARALLEL_WRITE";
    case SpDataAccessMode::COMMUTATIVE_WRITE: return "COMMUTATIVE_WRITE";
    case SpDataAccessMode::POTENTIAL_WRITE: return "POTENTIAL_WRITE";
    }
    return "UNDEFINED";
}

////////////////////////////////////////////////////////
/// Data mode => a mode + a data
////////////////////////////////////////////////////////

template <SpDataAccessMode AccessModeT, class HandleTypeT>
struct SpScalarDataMode{
    static_assert(std::is_reference<HandleTypeT>::value,
                  "The given type must be a reference");
public:
    // To test at compile time if it is a scalar
    static const bool IsScalar = true;
    // The access mode of the data/access pair
    static const SpDataAccessMode AccessMode = AccessModeT;
    // The original data type (including reference)
    using HandleTypeRef = HandleTypeT;
    static_assert(std::is_reference<HandleTypeRef>::value, "The given type must be a reference");
    // The original data type (without reference)
    using HandleTypeNoRef = std::remove_reference_t<HandleTypeRef>;
    static_assert(!std::is_reference<HandleTypeNoRef>::value, "HandleTypeNoRef should be without reference");
    // The original data type (without reference with pointer)
    using HandleTypePtr = std::remove_reference_t<HandleTypeRef>*;
    static_assert(!std::is_reference<HandleTypePtr>::value && std::is_pointer<HandleTypePtr>::value, "HandleTypeNoRef should be without reference");
    // The raw data type (no const not ref)
    using RawHandleType = std::remove_const_t<std::remove_reference_t<HandleTypeRef>>;
    static_assert(!std::is_reference<RawHandleType>::value && !std::is_const<RawHandleType>::value, "HandleTypeNoRef should be without reference");
protected:
    // The reference on the data
    HandleTypePtr ptrToData;

public:
    // Simple constructor (simply transfer the handle)
    constexpr explicit SpScalarDataMode(HandleTypeT& inHandle)
        : ptrToData(std::addressof(inHandle)){
    }

    SpScalarDataMode(const SpScalarDataMode&) = default;
    SpScalarDataMode(SpScalarDataMode&&) = delete;
    SpScalarDataMode& operator=(const SpScalarDataMode&) = delete;
    SpScalarDataMode& operator=(SpScalarDataMode&&) = delete;

    constexpr std::array<HandleTypePtr,1> getAllData(){
        return std::array<HandleTypePtr,1>{ptrToData};
    }

    constexpr HandleTypeRef getView(){
        return *ptrToData;
    }

    constexpr void updatePtr([[maybe_unused]] const long int position, HandleTypePtr ptr){
        assert(position < 1);
        ptrToData = ptr;
    }
};

template <SpDataAccessMode AccessModeT, class HandleTypeT, class AccessorTypeT>
struct SpContainerDataMode{
    static_assert(std::is_pointer<HandleTypeT>::value,
                  "The given type must be a pointer");
    static_assert(!std::is_reference<HandleTypeT>::value,
                  "The given type must be not a pointer");
public:
    // To test at compile time if it is a scalar
    static const bool IsScalar = false;
    // The access mode of the data/access pair
    static const SpDataAccessMode AccessMode = AccessModeT;
    // The original data type (without reference with pointer)
    using HandleTypePtr = HandleTypeT;
    static_assert(!std::is_reference<HandleTypePtr>::value && std::is_pointer<HandleTypePtr>::value, "HandleTypeNoRef should be without reference");
    // The original data type (without reference)
    using HandleTypeNoRef = std::remove_pointer_t<HandleTypePtr>;
    static_assert(!std::is_reference<HandleTypeNoRef>::value, "HandleTypeNoRef should be without reference");
    // The original data type (including reference)
    using HandleTypeRef = HandleTypeNoRef&;
    static_assert(std::is_reference<HandleTypeRef>::value, "The given type must be a reference");
    // The raw data type (no const not ref)
    using RawHandleType = std::remove_const_t<std::remove_reference_t<HandleTypeRef>>;
    static_assert(!std::is_reference<RawHandleType>::value && !std::is_const<RawHandleType>::value, "HandleTypeNoRef should be without reference");

    using AccessorType = AccessorTypeT;
private:
    AccessorTypeT accessor;
    small_vector<HandleTypePtr> ptrToData;

public:

    template <class VHC>
    SpContainerDataMode(HandleTypePtr inHandle, VHC&& inView)
        : accessor(inHandle, std::forward<VHC>(inView)){
        ptrToData.reserve(accessor.getSize());
        for(HandleTypePtr ptr : accessor){
            ptrToData.push_back(ptr);
        }
    }

    small_vector_base<HandleTypePtr>& getAllData(){
        return ptrToData;
    }

    AccessorType& getView(){
        return accessor;
    }

    void updatePtr(const long int position, HandleTypePtr ptr){
        assert(position < static_cast<long int>(ptrToData.size()));
        ptrToData[position] = ptr;
        accessor.updatePtr(position, ptr);
    }

    SpContainerDataMode(SpContainerDataMode&&) = default;
    SpContainerDataMode(const SpContainerDataMode&) = default;
    SpContainerDataMode& operator=(const SpContainerDataMode&) = delete;
    SpContainerDataMode& operator=(SpContainerDataMode&&) = delete;
};

////////////////////////////////////////////////////////
/// Access mode functions
////////////////////////////////////////////////////////

template <class DepType>
constexpr SpScalarDataMode<SpDataAccessMode::READ, const DepType&> SpRead(const DepType& inDep){
    return SpScalarDataMode<SpDataAccessMode::READ, const DepType&>(inDep);
}

template <class DepType>
constexpr SpScalarDataMode<SpDataAccessMode::WRITE, DepType&> SpWrite(DepType& inDep){
    static_assert(std::is_const<DepType>::value == false, "Write cannot be done on const value");
    return SpScalarDataMode<SpDataAccessMode::WRITE, DepType&>(inDep);
}

template <class DepType>
constexpr SpScalarDataMode<SpDataAccessMode::PARALLEL_WRITE, DepType&> SpParallelWrite(DepType& inDep){
    static_assert(std::is_const<DepType>::value == false, "Atomic Write cannot be done on const value");
    return SpScalarDataMode<SpDataAccessMode::PARALLEL_WRITE, DepType&>(inDep);
}

template <class DepType>
constexpr SpScalarDataMode<SpDataAccessMode::COMMUTATIVE_WRITE, DepType&> SpCommutativeWrite(DepType& inDep){
    static_assert(std::is_const<DepType>::value == false, "Commutative Write cannot be done on const value");
    return SpScalarDataMode<SpDataAccessMode::COMMUTATIVE_WRITE, DepType&>(inDep);
}

template <class DepType>
constexpr SpScalarDataMode<SpDataAccessMode::POTENTIAL_WRITE, DepType&> SpPotentialWrite(DepType& inDep){
    static_assert(std::is_const<DepType>::value == false, "Potential write cannot be done on const value");
    static_assert(SpDataCanBeDuplicate<DepType>::value, "Potentially written to data must be duplicatable");
    return SpScalarDataMode<SpDataAccessMode::POTENTIAL_WRITE, DepType&>(inDep);
}

////////////////////////////////////////////////////////

template <class DepType, class ViewType>
SpContainerDataMode<SpDataAccessMode::READ, const DepType*,SpArrayAccessor<const DepType>> SpReadArray(const DepType* inDep, ViewType&& inInterval){
    return SpContainerDataMode<SpDataAccessMode::READ, const DepType*,SpArrayAccessor<const DepType>>(inDep,std::forward<ViewType>(inInterval));
}

template <class DepType, class ViewType>
SpContainerDataMode<SpDataAccessMode::WRITE, DepType*,SpArrayAccessor<DepType>> SpWriteArray(DepType* inDep, ViewType&& inInterval){
    static_assert(std::is_const<DepType>::value == false, "SpWriteArray cannot be done on const value");
    return SpContainerDataMode<SpDataAccessMode::WRITE, DepType*,SpArrayAccessor<DepType>>(inDep,std::forward<ViewType>(inInterval));
}

template <class DepType, class ViewType>
SpContainerDataMode<SpDataAccessMode::COMMUTATIVE_WRITE, DepType*,SpArrayAccessor<DepType>> SpCommutativeWriteArray(DepType* inDep, ViewType&& inInterval){
    static_assert(std::is_const<DepType>::value == false, "SpCommutativeWriteArray cannot be done on const value");
    return SpContainerDataMode<SpDataAccessMode::COMMUTATIVE_WRITE, DepType*,SpArrayAccessor<DepType>>(inDep,std::forward<ViewType>(inInterval));
}

template <class DepType, class ViewType>
SpContainerDataMode<SpDataAccessMode::PARALLEL_WRITE, DepType*,SpArrayAccessor<DepType>> SpParallelWriteArray(DepType* inDep, ViewType&& inInterval){
    static_assert(std::is_const<DepType>::value == false, "SpParallelWriteArray cannot be done on const value");
    return SpContainerDataMode<SpDataAccessMode::PARALLEL_WRITE, DepType*,SpArrayAccessor<DepType>>(inDep,std::forward<ViewType>(inInterval));
}

template <class DepType, class ViewType>
SpContainerDataMode<SpDataAccessMode::POTENTIAL_WRITE, DepType*,SpArrayAccessor<DepType>> SpPotentialWriteArray(DepType* inDep, ViewType&& inInterval){
    static_assert(std::is_const<DepType>::value == false, "SpPotentialWriteArray cannot be done on const value");
    return SpContainerDataMode<SpDataAccessMode::POTENTIAL_WRITE, DepType*,SpArrayAccessor<DepType>>(inDep,std::forward<ViewType>(inInterval));
}

////////////////////////////////////////////////////////
/// Forbid on rvalue
////////////////////////////////////////////////////////

template <class DepType>
constexpr const DepType& SpRead(const DepType&& inDep) = delete;

template <class DepType>
constexpr std::remove_const_t<DepType>& SpWrite(DepType&& inDep) = delete;

template <class DepType>
constexpr std::remove_const_t<DepType>& SpParallelWrite(DepType&& inDep) = delete;

template <class DepType>
constexpr std::remove_const_t<DepType>& SpCommutativeWrite(DepType&& inDep) = delete;

template <class DepType>
constexpr std::remove_const_t<DepType>& SpPotentialWrite(DepType&& inDep) = delete;

////////////////////////////////////////////////////////
/// Get data from data, or SpDataAccessMode or pair
////////////////////////////////////////////////////////

template <SpDataAccessMode AccessMode, class ParamsType>
ParamsType& GetRealData(SpScalarDataMode<AccessMode, ParamsType>& scalarData){
    return scalarData.data;
}

template <SpDataAccessMode AccessMode, class HandleTypeRef, class ViewType>
ViewType& GetRealData(SpContainerDataMode<AccessMode, HandleTypeRef, ViewType>& containerData){
    return containerData.getView();
}

template<typename T, typename = void>
struct is_plus_equal_compatible : std::false_type
{ };

template<typename T>
struct is_plus_equal_compatible<T,
    typename std::enable_if<
        true,
        decltype(std::declval<T&>() += std::declval<const T&>(), (void)0)
        >::type
    > : std::true_type
{
};


////////////////////////////////////////////////////////
/// Test if a type has the getView method
////////////////////////////////////////////////////////

template <class T>
using has_getView_Test = decltype(std::declval<T>().getView());

template<class T>
using has_getView = SpUtils::detect<T, has_getView_Test>;

template <class T>
using has_getAllData_Test = decltype(std::declval<T>().getAllData());

template<class T>
using has_getAllData = SpUtils::detect<T, has_getAllData_Test>;

template <SpDataAccessMode dam1, SpDataAccessMode dam2>
struct access_modes_are_equal_internal : std::conditional_t<dam1==dam2, std::true_type, std::false_type> {};
    
template<SpDataAccessMode dam1, typename T, typename = std::void_t<>>
struct access_modes_are_equal : std::false_type {};

template <SpDataAccessMode dam1, typename T>
struct access_modes_are_equal<dam1, T, std::void_t<decltype(T::AccessMode)>> : access_modes_are_equal_internal<dam1, T::AccessMode> {};    
       
enum class SpCallableType {
    CPU=0,
    CUDA        
};

template <bool compileWithCuda, class T, SpCallableType ct>
class SpCallableWrapper {
private:
    using CallableTy = std::remove_reference_t<T>;
    CallableTy callable; 
public:
    static constexpr auto callable_type = ct;
    
    template <typename T2, typename=std::enable_if_t<std::is_same<std::remove_reference_t<T2>, CallableTy>::value>> 
    SpCallableWrapper(T2&& inCallable) : callable(std::forward<T2>(inCallable)) {}
       
    CallableTy& getCallableRef() {
        return callable;
    }
};

template <class T, SpCallableType ct>
class SpCallableWrapper<false, T, ct> {
public:
    template<typename T2>
    SpCallableWrapper(T2&&) {}
};

template <class T>
auto SpCpu(T &&callable) {
    return SpCallableWrapper<true, T, SpCallableType::CPU>(std::forward<T>(callable));
}

template <class T>
auto SpCuda(T&& callable) {
    return SpCallableWrapper<SpConfig::CompileWithCuda, T, SpCallableType::CUDA>(std::forward<T>(callable));
}

template <class T0>
struct is_instantiation_of_callable_wrapper : std::false_type {};

template <bool b, class T0, SpCallableType ct>
struct is_instantiation_of_callable_wrapper<SpCallableWrapper<b, T0, ct>> : std::true_type {};

template <class T0>
inline constexpr bool is_instantiation_of_callable_wrapper_v = is_instantiation_of_callable_wrapper<T0>::value;

template <class T0, SpCallableType callableType0>
struct is_instantiation_of_callable_wrapper_with_type : std::false_type {};

template <bool b, class T0, SpCallableType callableType0, SpCallableType callableType1>
struct is_instantiation_of_callable_wrapper_with_type<SpCallableWrapper<b, T0, callableType1>, callableType0> : 
std::conditional_t<callableType0==callableType1, std::true_type, std::false_type> {};

template <class T, SpCallableType callableType>
inline constexpr bool is_instantiation_of_callable_wrapper_with_type_v = is_instantiation_of_callable_wrapper_with_type<T, callableType>::value;

#endif
