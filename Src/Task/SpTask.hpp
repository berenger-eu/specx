///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPTASK_HPP
#define SPTASK_HPP

#include <tuple>
#include <unordered_map>
#include <typeinfo>
#include <type_traits>
#include <cstdlib>
#include <utility>

#include "SpAbstractTask.hpp"
#include "Data/SpDataHandle.hpp"
#include "Utils/SpUtils.hpp"
#include "Utils/small_vector.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#endif

class SpAbstractTaskGraph;

template <class RetType, class DataDependencyTupleTy, class CallableTupleTy>
class SpTask : public SpAbstractTaskWithReturn<RetType> {
    using Parent = SpAbstractTask;
    using ParentReturn = SpAbstractTaskWithReturn<RetType>;

    //! Number of parameters in the task function prototype
    static const long int NbParams = std::tuple_size<DataDependencyTupleTy>::value;

    //! Internal value for undefined dependences
    static constexpr long int UndefinedKey(){
        return -1;
    }

    //! Data handles
    std::array<SpDataHandle*,NbParams> dataHandles;
    //! Dependences' keys for each handle
    std::array<long int,NbParams> dataHandlesKeys;

    //! Extra handles
    small_vector<SpDataHandle*> dataHandlesExtra;
    //! Extra handles's dependences keys
    small_vector<long int> dataHandlesKeysExtra;

    //! DataDependency objects
    DataDependencyTupleTy tupleParams;
    
    //! Arguments for gpu callable
    std::array<std::pair<void*, std::size_t>, std::tuple_size_v<DataDependencyTupleTy>> gpuCallableArgs;
    
    //! Callables
    CallableTupleTy callables;

    ///////////////////////////////////////////////////////////////////////////////
    /// Methods to call the task function with a conversion from handle to data
    ///////////////////////////////////////////////////////////////////////////////

    //! Expand the tuple with the index and call getView
    template <class CallableTy, std::size_t... Is>
    static RetType SpTaskCoreWrapper(CallableTy& callable, DataDependencyTupleTy& dataDep, std::index_sequence<Is...>){
        if constexpr (is_instantiation_of_callable_wrapper_with_type_v<std::remove_reference_t<CallableTy>, SpCallableType::CPU>) {
            return std::invoke(callable.getCallableRef(), std::get<Is>(dataDep).getView()...);
        } else {
            return std::invoke(callable.getCallableRef(), std::get<Is>(dataDep).getGpuView()...);
        }
    }

    //! Dispatch use if RetType is not void (will set parent value with the return from function)
    template <class SRetType, class CallableTy>
    static void executeCore(SpAbstractTaskWithReturn<SRetType>* taskObject, CallableTy& callable, DataDependencyTupleTy& dataDep) {
        taskObject->setValue(SpTaskCoreWrapper(callable, dataDep, std::make_index_sequence<std::tuple_size<DataDependencyTupleTy>::value>{}));
    }

    //! Dispatch use if RetType is void (will not set parent value with the return from function)
    template <class CallableTy>
    static void executeCore(SpAbstractTaskWithReturn<void>* /*taskObject*/, CallableTy& callable, DataDependencyTupleTy& dataDep) {
        SpTaskCoreWrapper(callable, dataDep, std::make_index_sequence<NbParams>{});
    }
    
    void preTaskExecution(SpCallableType ct) final {
        std::size_t extraHandlesOffset = 0;
        
        SpUtils::foreach_in_tuple(
        [&, this](auto index, auto&& scalarOrContainerData) -> void {
            using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
            using TargetParamType = typename ScalarOrContainerType::RawHandleType;

            constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
            
            long int indexHh = 0;
            
            for([[maybe_unused]] typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                SpDataHandle* h = nullptr;
                
                if(indexHh == 0) {
                    h = dataHandles[index];
                } else {
                    h = dataHandlesExtra[extraHandlesOffset];
                    ++extraHandlesOffset;
                }
                
                switch(ct) {
                    case SpCallableType::CPU:
                    {
                        switch(h->getLocation()) {
                            case SpDataLocation::HOST:
                            {
                                // do nothing
                            }
                            break;
                            case SpDataLocation::DEVICE:
                            {
                                
                                if constexpr(std::is_trivially_copyable_v<TargetParamType>) {
                                    // cudaMemcpy Device -> Host
                                    std::memcpy(h->getRawPtr(), h->getDevicePtr(), sizeof(TargetParamType));
                                }
                                
                                if constexpr(accessMode != SpDataAccessMode::READ) {
                                    h->setDataLocation(SpDataLocation::HOST);
                                }
                            }
                            break;
                            case SpDataLocation::HOST_AND_DEVICE:
                            {
                                if constexpr(accessMode != SpDataAccessMode::READ) {
                                    h->setDataLocation(SpDataLocation::HOST);
                                }
                            }
                            break;
                            default:
                            break;
                        }
                    }
                    break;
                    case SpCallableType::GPU:
                    {
                        switch(h->getLocation()) {
                            case SpDataLocation::HOST:
                            {
                                if constexpr(std::is_trivially_copyable_v<TargetParamType>) {
                                    // cudaMemcpy Host -> Device
                                    void* devicePtr = std::malloc(sizeof(TargetParamType));
                                    
                                    std::memcpy(devicePtr, h->getRawPtr(), sizeof(TargetParamType));
                                    
                                    h->setDevicePtr(devicePtr);
                                } else {
                                    h->setDevicePtr(h->getRawPtr());
                                }
                                
                                if constexpr(accessMode != SpDataAccessMode::READ) {
                                    h->setDataLocation(SpDataLocation::DEVICE);
                                }
                            }
                            break;
                            case SpDataLocation::DEVICE:
                            {
                                // do nothing
                            }
                            break;
                            case SpDataLocation::HOST_AND_DEVICE:
                            {
                                if constexpr(accessMode != SpDataAccessMode::READ) {
                                    h->setDataLocation(SpDataLocation::DEVICE);
                                }
                            }
                            break;
                            default:
                            break;
                        }
                        
                        h->incrDeviceDataUseCount();
                        
                        gpuCallableArgs[indexHh] = {h->getDevicePtr(), sizeof(TargetParamType)};
                    }
                    break;
                    
                    default:
                    break;
                }
                
                ++indexHh;
            }
        }
        , this->getDataDependencyTupleRef());
    }

    //! Called by parent abstract task class
    void executeCore(SpCallableType ct) final {
        if constexpr(std::tuple_size_v<CallableTupleTy> == 1) {
                executeCore(this, std::get<0>(callables), tupleParams);
        } else {
            if(ct == SpCallableType::CPU) {
                executeCore(this, std::get<0>(callables), tupleParams);
            } else {
                executeCore(this, std::get<1>(callables), tupleParams);
            }
        }
    }
    
    void postTaskExecution(SpCallableType ct) final {
        // cudaStreamSynchronize(stream);
        
        std::size_t extraHandlesOffset = 0;
        
        SpUtils::foreach_in_tuple(
        [&, this](auto index, auto&& scalarOrContainerData) -> void {
            using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
            using TargetParamType = typename ScalarOrContainerType::RawHandleType;

            constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
            
            long int indexHh = 0;
            
            for([[maybe_unused]] typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                SpDataHandle* h = nullptr;
                
                if(indexHh == 0) {
                    h = dataHandles[index];
                } else {
                    h = dataHandlesExtra[extraHandlesOffset];
                    ++extraHandlesOffset;
                }
                
                switch(ct) {
                    case SpCallableType::CPU:
                    {
                        if constexpr(accessMode != SpDataAccessMode::READ) {
                            if(h->getDevicePtr() != nullptr) {
                                                                                              
                                if constexpr(std::is_trivially_copyable_v<TargetParamType>) {
                                    std::free(h->getDevicePtr());
                                }
                                
                                h->setDevicePtr(nullptr);
                            }
                        }
                    }
                    break;
                    case SpCallableType::GPU:
                    {
                        if constexpr(accessMode == SpDataAccessMode::READ) {
                            h->decrDeviceDataUseCount();
                            
                            if(h->getDeviceDataUseCount() == 0) {
                                // add h to unused handles
                            }
                        }
                    }
                    break;
                    
                    default:
                    break;
                }
                
                ++indexHh;
            }
        }
        , this->getDataDependencyTupleRef());
    }

public:
    //! Constructor from a task function
    template <typename... T>
    explicit SpTask(SpAbstractTaskGraph* const inAtg, const SpTaskActivation initialActivationState, 
                    const SpPriority &inPriority,
                    DataDependencyTupleTy &&inDataDepTuple,
                    CallableTupleTy &&inCallableTuple, T... t) 
        : SpAbstractTaskWithReturn<RetType>(inAtg, initialActivationState, inPriority),
        tupleParams(inDataDepTuple),
        callables(std::move(inCallableTuple)) {
        ((void) t, ...);
        std::fill_n(dataHandles.data(), NbParams, nullptr);
        std::fill_n(dataHandlesKeys.data(), NbParams, UndefinedKey());

#ifdef __GNUG__
        // if GCC then we ask for a clean type as default task name
        int status;
        char *demangledName = abi::__cxa_demangle(typeid(std::remove_reference_t<decltype(std::get<0>(callables))>).name(), 0, 0, &status);
        if(status == 0){
            assert(demangledName);
            Parent::setTaskName(demangledName);
        }
        else{
            Parent::setTaskName(typeid(std::remove_reference_t<decltype(std::get<0>(callables))>).name());
        }
        free(demangledName);
#else
        Parent::setTaskName(typeid(std::remove_reference_t<decltype(std::get<0>(callables))>).name());
#endif
    }

    DataDependencyTupleTy& getDataDependencyTupleRef() {
        return tupleParams;
    }

    //! Set the dependence at position HandleIdx in the prototype
    template <long int HandleIdx>
    void setDataHandle(SpDataHandle* inData, const long int inHandleKey){
        static_assert(HandleIdx < NbParams, "Cannot set more handle the NbParams");
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " setDataHandle " <<  HandleIdx << " at " << inData;
        assert(dataHandles[HandleIdx] == nullptr);
        dataHandles[HandleIdx] = inData;
        dataHandlesKeys[HandleIdx] = inHandleKey;
    }

    //! Add a dependence (extra), can be done at runtime
    template <long int HandleIdx>
    void addDataHandleExtra(SpDataHandle* inData, const long int inHandleKey){
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " addDataHandleExtra at " << inData;
        assert(dataHandles[HandleIdx] != nullptr);
        dataHandlesExtra.emplace_back(inData);
        dataHandlesKeysExtra.emplace_back(inHandleKey);
    }


    //! For speculation in order to update the memory pointer
    //! that will be used when the function will be called
    template <long int HandleIdx, class ParamType>
    void updatePtr(const long int position, ParamType* ptr){
        std::get<HandleIdx>(tupleParams).updatePtr(position, ptr);
    }

    //! The number of parameters in the task prototype
    long int getNbParams() final {
        return NbParams;
    }

    //! Return true if all dependences are ready
    //! Dependence are not use after this call, it is just a check
    bool dependencesAreReady() const final{
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            if(dataHandles[idxDeps]->canBeUsedByTask(this, dataHandlesKeys[idxDeps]) == false){
                SpDebugPrint() << "SpTask -- " << Parent::getId() << " dependencesAreReady FALSE, at index " << idxDeps << " " << dataHandles[idxDeps]
                                  << " address " << dataHandles[idxDeps]->template castPtr<int>() << "\n";
                return false;
            }
            SpDebugPrint() << "SpTask -- " << Parent::getId() << " dependencesAreReady TRUE, at index " << idxDeps << " " << dataHandles[idxDeps] << "\n";
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                if(dataHandlesExtra[idxDeps]->canBeUsedByTask(this, dataHandlesKeysExtra[idxDeps]) == false){
                SpDebugPrint() << "SpTask -- " << Parent::getId() << " dependencesAreReady FALSE, at index extra " << idxDeps << " " << dataHandlesExtra[idxDeps]
                                  << " address " << dataHandlesExtra[idxDeps]->template castPtr<int>() << "\n";
                    return false;
                }
            }
        }
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " dependencesAreReady TRUE";
        return true;
    }

    //! Tell the dependences that they are used by the current task
    void useDependences(std::unordered_set<SpDataHandle*>* exceptionList) final{
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " useDependences";
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            if((exceptionList == nullptr || exceptionList->find(dataHandles[idxDeps]) == exceptionList->end())){
                dataHandles[idxDeps]->setUsedByTask(this, dataHandlesKeys[idxDeps]);
                SpDebugPrint() << "SpTask -- " << Parent::getId() << " useDependences at index " << idxDeps << " " << dataHandles[idxDeps]
                                  << " address " << dataHandles[idxDeps]->template castPtr<int>() << "\n";
            }
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                if( exceptionList == nullptr || exceptionList->find(dataHandles[idxDeps]) == exceptionList->end()){
                    dataHandlesExtra[idxDeps]->setUsedByTask(this, dataHandlesKeysExtra[idxDeps]);
                    SpDebugPrint() << "SpTask -- " << Parent::getId() << " useDependences at index " << idxDeps << " " << dataHandlesExtra[idxDeps]
                                      << " address " << dataHandlesExtra[idxDeps]->template castPtr<int>() << "\n";
                }
            }
        }
    }

    //! Tell the dependences that they are no longer use by the current task,
    //! and fill with potential candidate.
    //! The algorithm will first release all the dependences, such that
    //! when filling with potentialReady we are able to find tasks that have more
    //! than one dependence in common with the current task.
    void releaseDependences(small_vector_base<SpAbstractTask*>* potentialReady) final {
        // Arrays of boolean flags indicating for each released dependency whether the "after release" pointed to
        // dependency slot in the corresponding data handle contains any unfullfilled memory access
        // requests.
        std::array<bool, NbParams> curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandles;
        small_vector<bool> curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandlesExtra(dataHandlesExtra.size());
        
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " releaseDependences";
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandles[idxDeps] = dataHandles[idxDeps]->releaseByTask(this, dataHandlesKeys[idxDeps]);
            SpDebugPrint() << "SpTask -- " << Parent::getId() << " releaseDependences FALSE, at index " << idxDeps << " " << dataHandles[idxDeps]
                              << " address " << dataHandles[idxDeps]->template castPtr<int>() << "\n";
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandlesExtra[idxDeps] = dataHandlesExtra[idxDeps]->releaseByTask(this, dataHandlesKeysExtra[idxDeps]);
                SpDebugPrint() << "SpTask -- " << Parent::getId() << " releaseDependences FALSE, at index " << idxDeps << " " << dataHandlesExtra[idxDeps]
                                  << " address " << dataHandlesExtra[idxDeps]->template castPtr<int>();
            }
        }

        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            if(curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandles[idxDeps]){
                dataHandles[idxDeps]->fillCurrentTaskList(potentialReady);
            }
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                if(curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandlesExtra[idxDeps]){
                    dataHandlesExtra[idxDeps]->fillCurrentTaskList(potentialReady);
                }
            }
        }
    }

    //! Fill with the tasks that depend on the current task
    virtual void getDependences(small_vector_base<SpAbstractTask*>* allDeps) const final {
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            dataHandles[idxDeps]->getDependences(allDeps, dataHandlesKeys[idxDeps]);
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                dataHandlesExtra[idxDeps]->getDependences(allDeps, dataHandlesKeysExtra[idxDeps]);
            }
        }
    }

    //! Fill with the tasks that current task depend on
    virtual void getPredecessors(small_vector_base<SpAbstractTask*>* allPredecessors) const final {
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            dataHandles[idxDeps]->getPredecessors(allPredecessors, dataHandlesKeys[idxDeps]);
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                dataHandlesExtra[idxDeps]->getPredecessors(allPredecessors, dataHandlesKeysExtra[idxDeps]);
            }
        }
    }

    //! Return true if at least one dependence is in mode inMode
    bool hasMode(const SpDataAccessMode inMode) const final{
        SpDebugPrint() << "SpTask -- " << Parent::getId() << " hasMode";
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            if(dataHandles[idxDeps]->getModeByTask(dataHandlesKeys[idxDeps]) == inMode){
                return true;
            }
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                if(dataHandlesExtra[idxDeps]->getModeByTask(dataHandlesKeysExtra[idxDeps]) == inMode){
                    return true;
                }
            }
        }
        return false;
    }

    small_vector<std::pair<SpDataHandle*,SpDataAccessMode>> getDataHandles() const final{
        small_vector<std::pair<SpDataHandle*,SpDataAccessMode>> data;
        data.reserve(NbParams + dataHandlesExtra.size());

        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            data.emplace_back(dataHandles[idxDeps], dataHandles[idxDeps]->getModeByTask(dataHandlesKeys[idxDeps]));
        }
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                data.emplace_back(dataHandlesExtra[idxDeps], dataHandlesExtra[idxDeps]->getModeByTask(dataHandlesKeysExtra[idxDeps]));
            }
        }
        return data;
    }
    
    //! Check that dependences are correct
    //! this is a simple check to ensure that no dependences are
    //! incompatible mode (like having a variable both in read and write)
    bool areDepsCorrect() const{
        std::unordered_map<SpDataHandle*,SpDataAccessMode> existingDeps;
        
        for(long int idxDeps = 0 ; idxDeps < NbParams ; ++idxDeps){
            assert(dataHandles[idxDeps]);
            assert(dataHandlesKeys[idxDeps] != UndefinedKey());
            auto testDep = existingDeps.find(dataHandles[idxDeps]);
            const SpDataAccessMode testMode = dataHandles[idxDeps]->getModeByTask(dataHandlesKeys[idxDeps]);
            if(testDep == existingDeps.end()){
                existingDeps[dataHandles[idxDeps]] = testMode;
            }
            else if(testMode == SpDataAccessMode::READ && testDep->second == SpDataAccessMode::READ){
            }
            else if(testMode == SpDataAccessMode::PARALLEL_WRITE && testDep->second == SpDataAccessMode::PARALLEL_WRITE){
            }
            else{
                return false;
            }
            existingDeps[dataHandles[idxDeps]] = testMode;
        }
        
        if(dataHandlesExtra.size()){
            for(long int idxDeps = 0 ; idxDeps < static_cast<long int>(dataHandlesExtra.size()) ; ++idxDeps){
                auto testDep = existingDeps.find(dataHandlesExtra[idxDeps]);
                const SpDataAccessMode testMode = dataHandlesExtra[idxDeps]->getModeByTask(dataHandlesKeysExtra[idxDeps]);
                if(testDep == existingDeps.end()){
                    existingDeps[dataHandlesExtra[idxDeps]] = testMode;
                }
                else if(testMode == SpDataAccessMode::READ && testDep->second == SpDataAccessMode::READ){
                }
                else if(testMode == SpDataAccessMode::PARALLEL_WRITE && testDep->second == SpDataAccessMode::PARALLEL_WRITE){
                }
                else{
                    return false;                
                }
                existingDeps[dataHandlesExtra[idxDeps]] = testMode;
            }
        }
        
        return true;
    }

    bool hasCallableOfType([[maybe_unused]] const SpCallableType sct) const override final {
        if constexpr(std::tuple_size_v<CallableTupleTy> == 1) {
            return std::tuple_element_t<0, CallableTupleTy>::callable_type == sct; 
        } else {
            return true;
        }
    }
    
    std::string getTaskBodyString() override {
        
        std::ostringstream os;
        
        for(size_t i=0; i < NbParams; i++) {
            os << SpModeToStr(dataHandles[i]->getModeByTask(dataHandlesKeys[i])) << " " << dataHandles[i]->getRawPtr() << std::endl;
        }
        
        for(size_t i=0; i < dataHandlesExtra.size(); i++) {
            os << SpModeToStr(dataHandlesExtra[i]->getModeByTask(dataHandlesKeysExtra[i])) << " " << dataHandlesExtra[i]->getRawPtr() << std::endl;
        }
        return os.str();
    }
};

template <class RetType, class DataDependencyTupleTy, class CallableTupleTy>
class SpSelectTask : public SpTask<RetType, DataDependencyTupleTy, CallableTupleTy>
{
    using Parent = SpTask<RetType, DataDependencyTupleTy, CallableTupleTy>;
    
    // flag indicating if the select task is carrying surely written values over
    bool isCarrSurWrittValuesOver;
    
public:
    explicit SpSelectTask(SpAbstractTaskGraph* const inAtg, const SpTaskActivation initialActivationState, 
                          const SpPriority& inPriority,
                          DataDependencyTupleTy &&inDataDepTuple,
                          CallableTupleTy &&inCallableTuple, bool iCSWVO)
                        : Parent(inAtg, initialActivationState, inPriority,
                          std::move(inDataDepTuple), std::move(inCallableTuple)), isCarrSurWrittValuesOver(iCSWVO) {}
    
    void setEnabledDelegate(const SpTaskActivation inIsEnable) override final {
        if((inIsEnable == SpTaskActivation::DISABLE && !isCarrSurWrittValuesOver)
            || inIsEnable == SpTaskActivation::ENABLE) {
            this->setEnabled(inIsEnable);
        }
    }
};

#endif
