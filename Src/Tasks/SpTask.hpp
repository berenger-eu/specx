///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPTASK_HPP
#define SPTASK_HPP

#include <tuple>
#include <unordered_map>

#include "SpAbstractTask.hpp"
#include "Data/SpDataHandle.hpp"
#include "Utils/SpUtils.hpp"

#ifdef __GNUG__
#include <cxxabi.h>
#endif

template <class TaskFuncType, class RetType, class ... Params>
class SpTask : public SpAbstractTaskWithReturn<RetType> {
    using Parent = SpAbstractTask;
    using ParentReturn = SpAbstractTaskWithReturn<RetType>;
    using TupleParamsType = std::tuple<Params...>;

    //! Number of parameters in the task function prototype
    static const long int NbParams = sizeof...(Params);

    //! Internal value for undefined dependences
    static constexpr long int UndefinedKey(){
        return -1;
    }

    //! Function to call
    TaskFuncType taskCallback;

    //! Data handles
    std::array<SpDataHandle*,NbParams> dataHandles;
    //! Dependences' keys for each handle
    std::array<long int,NbParams> dataHandlesKeys;

    //! Extra handles
    std::vector<SpDataHandle*> dataHandlesExtra;
    //! Extra handles's dependences keys
    std::vector<long int> dataHandlesKeysExtra;

    //! Params (inside data mode)
    TupleParamsType tupleParams;

    ///////////////////////////////////////////////////////////////////////////////
    /// Methods to call the task function with a conversion from handle to data
    ///////////////////////////////////////////////////////////////////////////////

    //! Expand the tuple with the index and call getView
    template <std::size_t... Is>
    static RetType SpTaskCoreWrapper(TaskFuncType& taskCallback, TupleParamsType& params, std::index_sequence<Is...>){
        return taskCallback( std::get<Is>(params).getView() ... );
    }

    //! Dispatch use if RetType is not void (will set parent value with the return from function)
    template <class SRetType>
    static void executeCore(SpAbstractTaskWithReturn<SRetType>* taskObject, TaskFuncType& taskCallback, TupleParamsType& params) {
        taskObject->setValue(SpTaskCoreWrapper(taskCallback, params, std::make_index_sequence<std::tuple_size<TupleParamsType>::value>{}));
    }

    //! Dispatch use if RetType is void (will not set parent value with the return from function)
    static void executeCore(SpAbstractTaskWithReturn<void>* /*taskObject*/, TaskFuncType& taskCallback, TupleParamsType& params) {
        SpTaskCoreWrapper(taskCallback, params, std::make_index_sequence<std::tuple_size<TupleParamsType>::value>{});
    }

    //! Called by parent abstract task class
    void executeCore() final {
        //Equivalent to ParentReturn::setValue(SpTaskCoreWrapper(taskCallback, dataHandles.data()));
        executeCore(this, taskCallback, tupleParams);
    }

public:
    //! Constructor from a task function
    template <class TaskFuncTypeCstr, typename... T>
    explicit SpTask(TaskFuncTypeCstr&& inTaskCallback, const SpPriority& inPriority,
                   TupleParamsType&& inTupleParams, T... t) 
        : SpAbstractTaskWithReturn<RetType>(inPriority), taskCallback(std::forward<TaskFuncTypeCstr>(inTaskCallback)),
          tupleParams(inTupleParams){
        ((void)t, ...);
        std::fill_n(dataHandles.data(), NbParams, nullptr);
        std::fill_n(dataHandlesKeys.data(), NbParams, UndefinedKey());

#ifdef __GNUG__
        // if GCC then we ask for a clean type as default task name
        int status;
        Parent::setTaskName(abi::__cxa_demangle(typeid(TaskFuncType).name(), 0, 0, &status));
#else
        Parent::setTaskName(typeid(TaskFuncType).name());
#endif
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
    void releaseDependences(std::vector<SpAbstractTask*>* potentialReady) final {
        // Arrays of boolean flags indicating for each released dependency whether the "after release" pointed to
        // dependency slot in the corresponding data handle contains any unfullfilled memory access
        // requests.
        std::array<bool, NbParams> curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandles;
        std::vector<bool> curPoinToDepSlotContainsAnyUnfulMemoryAccReqDataHandlesExtra(dataHandlesExtra.size());
        
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
    virtual void getDependences(std::vector<SpAbstractTask*>* allDeps) const final {
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
    virtual void getPredecessors(std::vector<SpAbstractTask*>* allPredecessors) const {
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

    std::vector<std::pair<SpDataHandle*,SpDataAccessMode>> getDataHandles() const final{
        std::vector<std::pair<SpDataHandle*,SpDataAccessMode>> data;
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
            else if(testMode == SpDataAccessMode::ATOMIC_WRITE && testDep->second == SpDataAccessMode::ATOMIC_WRITE){
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
                else if(testMode == SpDataAccessMode::ATOMIC_WRITE && testDep->second == SpDataAccessMode::ATOMIC_WRITE){
                }
                else{
                    return false;                
                }
                existingDeps[dataHandlesExtra[idxDeps]] = testMode;
            }
        }
        
        return true;
    }
};

template <class TaskFuncType, class RetType, class ... Params>
class SpSelectTask : public SpTask<TaskFuncType, RetType, Params...>
{
    using Parent = SpTask<TaskFuncType, RetType, Params...>;
    using TupleParamsType = std::tuple<Params...>;
    
    // flag indicating if the select task is carrying surely written values over
    bool isCarrSurWrittValuesOver;
    
public:
    template <class TaskFuncTypeCstr, typename... T>
    explicit SpSelectTask(TaskFuncTypeCstr&& inTaskCallback, const SpPriority& inPriority,
                          TupleParamsType&& inTupleParams, bool iCSWVO)
                        : Parent(std::forward<TaskFuncTypeCstr>(inTaskCallback), inPriority,
                          std::forward<TupleParamsType>(inTupleParams)), isCarrSurWrittValuesOver(iCSWVO) {}
    
    void setEnabledDelegate(const SpTaskActivation inIsEnable) override final {
        if((inIsEnable == SpTaskActivation::DISABLE && !isCarrSurWrittValuesOver)
            || inIsEnable == SpTaskActivation::ENABLE) {
            this->setEnabled(inIsEnable);
        }
    }
};

#endif
