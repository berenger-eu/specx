///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPRUNTIME_HPP
#define SPRUNTIME_HPP

#include <functional>
#include <list>
#include <map>
#include <unordered_map>
#include <memory>
#include <unistd.h>
#include <cmath>
#include <type_traits>
#include <initializer_list>

#include "Utils/SpUtils.hpp"
#include "Tasks/SpAbstractTask.hpp"
#include "Tasks/SpTask.hpp"
#include "SpDependence.hpp"
#include "Data/SpDataHandle.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpPriority.hpp"
#include "Utils/SpProbability.hpp"
#include "Utils/SpTimePoint.hpp"
#include "Random/SpMTGenerator.hpp"
#include "Schedulers/SpTasksManager.hpp"
#include "Output/SpDotDag.hpp"
#include "Output/SpSvgTrace.hpp"
#include "Speculation/SpSpecTaskGroup.hpp"
#include "Speculation/SpSpeculativeModel.hpp"

//! The runtime is the main component of spetabaru.
template <SpSpeculativeModel SpecModel = SpSpeculativeModel::SP_MODEL_1>
class SpRuntime : public SpAbstractToKnowReady {
    template <class Tuple, std::size_t IdxData>
    static void CheckParam(){
        using ParamType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        static_assert(has_getView<ParamType>::value, "Converted object to a task must have getView method");
        static_assert(has_getAllData<ParamType>::value, "Converted object to a task must have getAllData method");
    }

    template <class Tuple, std::size_t... Is>
    static auto CheckPrototypeCore(std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value >= 1, "You must pass at least a function to create a task");

        ((void) CheckParam<Tuple, Is>(), ...);

        using TaskCore = typename std::remove_reference<typename std::tuple_element<std::tuple_size<Tuple>::value-1, Tuple>::type>::type;
        static_assert(std::is_invocable<TaskCore, decltype(std::declval<typename std::tuple_element<Is, Tuple>::type>().getView()) ...>::value,
                      "Impossible to invoc le last argument passing all the previous ones");
    }

     static_assert(SpecModel == SpSpeculativeModel::SP_MODEL_1
                      || SpecModel == SpSpeculativeModel::SP_MODEL_2
                      || SpecModel == SpSpeculativeModel::SP_MODEL_3, "Should not happen");

    //! Threads
    std::vector<std::thread> threads;
    //! Number of threads
    const int nbThreads;

    //! All data handles
    std::unordered_map<void*, std::unique_ptr<SpDataHandle> > allDataHandles;

    //! Creation time point
    SpTimePoint startingTime;

    //! Internal scheduler of tasks
    SpTasksManager scheduler;

    ///////////////////////////////////////////////////////////////////////////
    /// Data management part
    ///////////////////////////////////////////////////////////////////////////

    template <class ParamsType>
    SpDataHandle* getDataHandleCore(ParamsType& inParam){
        auto iterHandleFound = allDataHandles.find(const_cast<std::remove_const_t<ParamsType>*>(&inParam));
        if(iterHandleFound == allDataHandles.end()){
            SpDebugPrint() << "SpRuntime -- => Not found";
            // Create a new data handle
            auto newHandle = std::make_unique<SpDataHandle>(const_cast<std::remove_const_t<ParamsType>*>(&inParam));
            // Save the ptr
            SpDataHandle* newHandlePtr = newHandle.get();
            // Move in the map
            allDataHandles[const_cast<std::remove_const_t<ParamsType>*>(&inParam)] = std::move(newHandle);
            // Return the ptr
            return newHandlePtr;
        }
        SpDebugPrint() << "SpRuntime -- => Found";
        // Exists, return the pointer
        return iterHandleFound->second.get();
    }

    template <class ParamsType>
    typename std::enable_if<ParamsType::IsScalar,std::array<SpDataHandle*,1>>::type
    getDataHandle(ParamsType& scalarData){
        static_assert(std::tuple_size<decltype(scalarData.getAllData())>::value, "Size must be one for scalar");
        typename ParamsType::HandleTypePtr ptrToObject = scalarData.getAllData()[0];
        SpDebugPrint() << "SpRuntime -- Look for " << ptrToObject << " with mode provided " << SpModeToStr(ParamsType::AccessMode)
                       << " scalarData address " << &scalarData;
        return std::array<SpDataHandle*,1>{getDataHandleCore(*ptrToObject)};
    }

    template <class ParamsType>
    typename std::enable_if<!ParamsType::IsScalar,std::vector<SpDataHandle*>>::type
    getDataHandle(ParamsType& containerData){
        std::vector<SpDataHandle*> handles;
        for(typename ParamsType::HandleTypePtr ptrToObject : containerData.getAllData()){
            SpDebugPrint() << "SpRuntime -- Look for " << ptrToObject << " with array view in getDataHandleExtra";
            SpDataHandle* handle = getDataHandleCore(*ptrToObject);
            handles.emplace_back(handle);
        }
        return handles;
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Speculation starts here
    ///////////////////////////////////////////////////////////////////////////

    class SpAbstractDeleter{
    public:
        virtual ~SpAbstractDeleter(){}
        virtual void deleteObject(void* ptr) = 0;
    };

    template <class ObjectType>
    class SpDeleter : public SpAbstractDeleter{
    public:
        ~SpDeleter() = default;

        void deleteObject(void* ptr) override final{
            delete reinterpret_cast<ObjectType*>(ptr);
        }
    };

    struct SpCurrentCopy{
        SpCurrentCopy()
            :originAdress(nullptr), sourceAdress(nullptr), latestAdress(nullptr), lastestSpecGroup(nullptr),
              latestCopyTask(nullptr), usedInRead(false){
        }

        bool isDefined() const{
            return originAdress != nullptr;
        }

        void* originAdress;
        void* sourceAdress;
        void* latestAdress;
        SpGeneralSpecGroup<SpecModel>* lastestSpecGroup;
        SpAbstractTask* latestCopyTask;
        bool usedInRead;

        std::shared_ptr<SpAbstractDeleter> deleter;
    };
    
    using CopyMapTy = std::unordered_map<const void*, SpCurrentCopy>;
    using CopyMapPtrTy = CopyMapTy*;
    
    std::vector<CopyMapTy> copyMaps;
    CopyMapTy emptyCopyMap;
    SpGeneralSpecGroup<SpecModel>* currentSpecGroup;
    std::list<std::unique_ptr<SpGeneralSpecGroup<SpecModel>>> specGroups;
    std::mutex specGroupMutex;

    void releaseCopies(){
        for(auto& copyMapIt : copyMaps){
            for(auto &iter : copyMapIt) {
                assert(iter.second.latestAdress);
                assert(iter.second.deleter);
                iter.second.deleter->deleteObject(iter.second.latestAdress);
            }
            copyMapIt.clear();
        }
        copyMaps.clear();
    }
    
    template <class Tuple, std::size_t IdxData, class TaskCorePtr, std::size_t N>
    inline void coreHandleCreation(Tuple& args, TaskCorePtr& aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
        coreHandleCreationAux<false, Tuple, IdxData, TaskCorePtr>(args, aTask, copyMapsToLookInto);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Core part of the task creation
    ///////////////////////////////////////////////////////////////////////////
    
    //! Convert tuple to data and call the function
    //! Args is a value to allow for move or pass a rvalue reference
    template <template<typename...> typename TaskType, const bool isSpeculative, class Tuple, std::size_t... Is, typename... T, std::size_t N>
    auto coreTaskCreationAux(const SpTaskActivation inActivation, const SpPriority& inPriority, Tuple args, std::index_sequence<Is...>,
                             const std::array<CopyMapPtrTy, N>& copyMapsToLookInto, T... additionalArgs){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        SpDebugPrint() << "SpRuntime -- coreTaskCreation";

        // Get the type of the function (maybe class, lambda, etc.)
        using TaskCore = typename std::remove_reference<typename std::tuple_element<std::tuple_size<Tuple>::value-1, Tuple>::type>::type;

        // Get the task object
        TaskCore taskCore = std::move(std::get<(std::tuple_size<Tuple>::value-1)>(args));

        // Get the return type of the task (can be void)
        using RetType = decltype(taskCore(std::get<Is>(args).getView()...));
        
        using TaskTy = TaskType<TaskCore, RetType, typename std::remove_reference_t<typename std::tuple_element<Is, Tuple>::type> ... >;
        
        // Create a task with a copy of the args
        auto aTask = new TaskTy(std::move(taskCore), inActivation, inPriority, std::make_tuple(std::get<Is>(args)...), additionalArgs...);

        // Lock the task
        aTask->takeControl();
        
        // Add the handles
        if constexpr(!isSpeculative) {
            ((void) coreHandleCreation<Tuple, Is, decltype(aTask)>(args, aTask, copyMapsToLookInto), ...);
        } else {
            ((void) coreHandleCreationSpec<Tuple, Is, decltype(aTask)>(args, aTask, copyMapsToLookInto), ...);
        }
        // Check coherency
        assert(aTask->areDepsCorrect());

        // The task has been initialized
        aTask->setState(SpTaskState::INITIALIZED);

        // Get the view
        typename SpAbstractTaskWithReturn<RetType>::SpTaskViewer descriptor = aTask->getViewer();

        aTask->setState(SpTaskState::WAITING_TO_BE_READY);
        
        if(currentSpecGroup){
            currentSpecGroup->addCopyTask(aTask);
            aTask->setSpecGroup(currentSpecGroup);
        }
        
        aTask->releaseControl();

        SpDebugPrint() << "SpRuntime -- coreTaskCreation => " << aTask << " of id " << aTask->getId();
        
        // Push to the scheduler
        scheduler.addNewTask(aTask);
        
        // Return the view
        return descriptor;
    }
    
    template <class Tuple, std::size_t... Is, std::size_t N>
    inline auto coreTaskCreation(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto, 
                                 const SpTaskActivation inActivation, 
                                 const SpPriority& inPriority, 
                                 Tuple args, 
                                 std::index_sequence<Is...> is){
            return coreTaskCreationAux<SpTask, false>(inActivation, inPriority, args, is, copyMapsToLookInto);
    }
    
    //! Convert tuple to data and call the function
    template <class Tuple, std::size_t... Is, std::size_t N>
    inline auto coreTaskCreationSpeculative(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                            const SpTaskActivation inActivation,
                                            const SpPriority& inPriority,
                                            Tuple& args,
                                            std::index_sequence<Is...> is) {
        return coreTaskCreationAux<SpTask, true>(inActivation, inPriority, args, is, copyMapsToLookInto);
    }

    static std::vector<SpAbstractTask*> copyMapToTaskVec(const std::unordered_map<const void*, SpCurrentCopy>& map){
        std::vector<SpAbstractTask*> list;
        list.reserve(map.size());
        for( auto const& cp : map ) {
            list.emplace_back(cp.second.latestCopyTask);
        }
        return list;
    }
    
    template <const bool isPotentialTask, class... ParamsAndTask>
    inline auto preCoreTaskCreationAux(const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params) {
        scheduler.lockListenersReadyMutex();
        specGroupMutex.lock();
        std::shared_ptr<std::atomic<size_t>> numberOfSiblingSpeculativeTasksCounter = std::make_shared<std::atomic<size_t>>(0);
        auto res = preCoreTaskCreationAuxRec<isPotentialTask>(copyMaps.begin(), copyMaps.end(), numberOfSiblingSpeculativeTasksCounter, false, inPriority, inProbability, params...);
        specGroupMutex.unlock();
        scheduler.unlockListenersReadyMutex();
        return res;
    }
    
    template <typename T>
    void addCallbackToTask(T& inTaskView, SpGeneralSpecGroup<SpecModel> *inSg) {
        inTaskView.addCallback([this, aTaskPtr = inTaskView.getTaskPtr(), specGroupPtr = inSg]
                                (const bool alreadyDone, const bool& taskRes, SpAbstractTaskWithReturn<bool>::SpTaskViewer& /*view*/,
                                const bool isEnabled){
                                    if(isEnabled){
                                        if(!alreadyDone){
                                            assert(SpUtils::GetThreadId() != 0);
                                            specGroupMutex.lock();
                                        }
                                        specGroupPtr->setSpeculationCurrentResult(!taskRes);
                                        if(!alreadyDone){
                                            assert(SpUtils::GetThreadId() != 0);
                                            specGroupMutex.unlock();
                                        }
                                    }
                                });
    }
    
    template <const bool isPotentialTask, class... ParamsAndTask>
    inline auto preCoreTaskCreationIterativeWrapper(const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params) {
        scheduler.lockListenersReadyMutex();
        specGroupMutex.lock();
        
        auto res = preCoreTaskCreationIterative(inPriority, inProbability, params...);
        
        specGroupMutex.unlock();
        scheduler.unlockListenersReadyMutex();
        
        return res;
    }
    template <const bool isPotentialTask, class... ParamsAndTask>
    inline auto preCoreTaskCreationIterative(const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params) {
        
        static_assert(SpecModel == SpSpeculativeModel::SP_MODEL_1
                      || SpecModel == SpSpeculativeModel::SP_MODEL_2
                      || SpecModel == SpSpeculativeModel::SP_MODEL_3, "Should not happen");

		auto tuple = std::forward_as_tuple(params...);
		auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        
        auto it = copyMaps.begin();
        
        if constexpr (!isPotentialTask && allAreCopiableAndDeleteable<decltype (tuple)>(sequenceParamsNoFunction) == false){
            for(; it != copyMaps.end(); it++){
                manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                removeAllCorrespondingCopies(*it, tuple, sequenceParamsNoFunction);
            }
            return coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                    SpTaskActivation::ENABLE, inPriority, std::move(tuple), sequenceParamsNoFunction);
        } else {
            static_assert(allAreCopiableAndDeleteable<decltype(tuple)>(sequenceParamsNoFunction) == true, "Add data passed to a potential task must be copiable");
            
            using TaskViewTy = decltype(coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, std::declval<SpTaskActivation>(), inPriority, tuple, sequenceParamsNoFunction));
            
            std::vector<TaskViewTy> speculativeTasks;
            std::vector<std::function<void()>> selectTaskCreationFunctions;
            
            std::shared_ptr<std::atomic<size_t>> numberOfSpeculativeSiblingSpecGroupsCounter = std::make_shared<std::atomic<size_t>>(0);
            
            for(; it != copyMaps.end(); ++it) {
                
                std::unordered_map<const void*, SpCurrentCopy> l1, l2, l1p;
                
                auto groups = getCorrespondingCopyGroups(*it, tuple, sequenceParamsNoFunction);
                bool oneGroupDisableOrFailed = false;
                
                for(auto gp : groups){
                    if(gp->isSpeculationDisable() || gp->didSpeculationFailed() || gp->didParentSpeculationFailed()){
                        oneGroupDisableOrFailed = true;
                        break;
                    }
                }
                
                const bool taskAlsoSpeculateOnOther = (groups.size() != 0 && !oneGroupDisableOrFailed);
                
                if(!taskAlsoSpeculateOnOther) {
                    manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                    removeAllCorrespondingCopies(*it, tuple, sequenceParamsNoFunction);
                } else {
                    
                    (*numberOfSpeculativeSiblingSpecGroupsCounter)++;
                    std::unique_ptr<SpGeneralSpecGroup<SpecModel>> specGroupSpecTask = std::make_unique<SpGeneralSpecGroup<SpecModel>>(true, numberOfSpeculativeSiblingSpecGroupsCounter);
                    specGroupSpecTask->addParents(groups);
                    
                    if constexpr(isPotentialTask) {
                        currentSpecGroup = specGroupSpecTask.get();
                        l1 = copyIfMaybeWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)}, specGroupSpecTask->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        assert(taskAlsoSpeculateOnOther == true || l1.size());
                        currentSpecGroup = nullptr;
                    }
                    
                    currentSpecGroup = specGroupSpecTask.get();
                    l2 = copyIfWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)}, specGroupSpecTask->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                    currentSpecGroup = nullptr;
                    
                    if constexpr(isPotentialTask && SpecModel != SpSpeculativeModel::SP_MODEL_2) {
                        l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)}, specGroupSpecTask->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                    }
                    
                    TaskViewTy taskViewSpec = coreTaskCreationSpeculative(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)}, specGroupSpecTask->getActivationStateForSpeculativeTask(), 
                                                                        inPriority, tuple, sequenceParamsNoFunction);
                    specGroupSpecTask->setSpecTask(taskViewSpec.getTaskPtr());
                    taskViewSpec.getTaskPtr()->setSpecGroup(specGroupSpecTask);
                    speculativeTasks.push_back(taskViewSpec);
                    
                    std::vector<std::function<void()>> functions = createSelectTaskCreationFunctions(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)}, specGroupSpecTask.get(), inPriority, tuple, sequenceParamsNoFunction);
                    
                    selectTaskCreationFunctions.reserve(selectTaskCreationFunctions.size() + functions.size());
                    selectTaskCreationFunctions.insert(selectTaskCreationFunctions.end(), functions.begin(), functions.end());
                    
                    manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                    
                    specGroups.emplace_back(std::move(specGroupSpecTask));
                    
                    if constexpr(isPotentialTask && SpecModel != SpSpeculativeModel::SP_MODEL_2) {
                        it->merge(l1p);
                    }
                }
            }
                
            if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                if(copyMaps.empty()) {
                    it = copyMaps.emplace_back(copyMaps.end());
                } else {
                    it = copyMaps.begin();
                }
            }else {
                it = copyMaps.emplace(copyMaps.end());
            }
            
            std::unique_ptr<SpGeneralSpecGroup<SpecModel>> specGroupNormalTask = std::make_unique<SpGeneralSpecGroup<SpecModel>>(!speculativeTasks.empty(), numberOfSpeculativeSiblingSpecGroupsCounter);
            
            currentSpecGroup = specGroupNormalTask.get();
            if constexpr(isPotentialTask) {
                std::unordered_map<const void*, SpCurrentCopy> l1p;
                if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_1) {
                    if(speculativeTasks.empty()) {
                        l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, specGroupNormalTask->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        it->merge(l1p);
                    }
                }else{
                    l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, specGroupNormalTask->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                    it->merge(l1p);
                }
                currentSpecGroup = nullptr;
            }
            
            TaskViewTy result = coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, specGroupNormalTask->getActivationStateForMainTask(), inPriority, tuple, sequenceParamsNoFunction);
            
            specGroupNormalTask->setMainTask(result);
            result.getTaskPtr()->setSpecGroup(specGroupNormalTask);
            
            if(isPotentialTask) {
            
                for(auto &t : speculativeTasks) {
                    SpGeneralSpecGroup<SpecModel>*sg = t.getTaskPtr()->template getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
                    sg->setMainTask(result);
                }
                
                for(auto &f : selectTaskCreationFunctions) {
                    f();
                }
                
                for(auto &t : speculativeTasks) {
                    SpGeneralSpecGroup<SpecModel>*sg = t.getTaskPtr()->template getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
                    addCallbackToTask(t, sg);
                }
                
                addCallbackToTask(result, specGroupNormalTask);
            
            }
            
            specGroups.emplace_back(std::move(specGroupNormalTask));
            
            return result;
        }
    }
    
    template <class Tuple, std::size_t IdxData, std::size_t N>
    std::vector<std::function<void()>> coreCreateSelectTaskCreationFunctions(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                   SpGeneralSpecGroup<SpecModel> *sg,
                                                   const SpPriority& inPriority,
                                                   Tuple& args){
        using ScalarOrContainerType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        using TargetParamType = typename ScalarOrContainerType::RawHandleType;
        static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                      "They should all be default constructible here");
                      
        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;

        std::vector<std::function<void()>> res;

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            SpDataHandle* h1 = hh[indexHh];
            for(auto me : copyMapsToLookInto) {
                if(auto found = me->find(ptr); found != me->end()){
                    const SpCurrentCopy& cp = found->second;
                    assert(cp.isDefined());
                    assert(cp.originAdress == ptr);
                    assert(cp.latestAdress != nullptr);

                    assert(accessMode == SpDataAccessMode::READ || found->second.usedInRead == false);
                    
                    if(accessMode != SpDataAccessMode::READ){
                        bool isCarryingSurelyWrittenValuesOver = accessMode == SpDataAccessMode::WRITE;
                        void* cpLatestAddress = cp.latestAdress;
                        
                        auto s = [this, isCarryingSurelyWrittenValuesOver, inPriority, h1, cpLatestAddress, sg]() {
                        
                            SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cpLatestAddress));
                            
                            auto taskViewSelect = this->taskInternalSpSelect(
                                               emptyCopyMap,
                                               isCarryingSurelyWrittenValuesOver,
                                               sg->getActivationStateForSelectTask(isCarryingSurelyWrittenValuesOver),
                                               inPriority,
                                               SpWrite(*h1->castPtr<TargetParamType>()),
                                               SpWrite(*h1copy->castPtr<TargetParamType>()),
                                               [](TargetParamType& output, TargetParamType& input){
                                                   output = std::move(input);
                                               }
                            );
                            
                            taskViewSelect.setTaskName("sp-select");
                            taskViewSelect.getTaskPtr()->setSpecGroup(sg);
                            sg->addSelectTask(taskViewSelect.getTaskPtr());
                            
                            // delete copied data carried over by select task
                            auto taskViewDelete = taskInternal(emptyCopyMap, SpTaskActivation::ENABLE, SpPriority(0),
                                         SpWrite(*reinterpret_cast<TargetParamType*>(cpLatestAddress)),
                                         [](TargetParamType& output){
                                             delete &output;
                                         });
                            
                            taskViewDelete.setTaskName("sp-delete");
                            taskViewDelete.getTaskPtr()->setSpecGroup(sg);
                        };
                        
                        res.push_back(s);
                        
                        // delete copy from copy map
                        me->erase(found);
                    }
                    else{
                       found->second.usedInRead = true;
                    }
                    
                    break;
                }
                
            }
            
            indexHh += 1;
        }
        
        return res;
    }
    
    
    template <class Tuple, std::size_t... Is, std::size_t N>
    std::vector<std::function<void()>> createSelectTaskCreationFunctions(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                               SpGeneralSpecGroup<SpecModel> *sg,
                                               const SpPriority& inPriority,
                                               Tuple& args,
                                               std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        
        std::vector<std::function<void()>> res;
        
        if constexpr(sizeof...(Is) > 0) {
            ([](std::vector<std::function<void()>> &l, std::vector<std::function<void()>>&& l2) {
                l.reserve(l.size() + l2.size());
                l.insert(l.end(), l2.begin(), l2.end());
            }(res, coreCreateSelectTaskCreationFunctions<Tuple, Is>(copyMapsToLookInto, sg, inPriority, args)), ...);
        }
        
        return res;
    }
    
    template <const bool isPotentialTask, class... ParamsAndTask>
    auto preCoreTaskCreationAuxRec(typename std::vector<std::unordered_map<const void*, SpCurrentCopy>>::iterator it,
                                   typename std::vector<std::unordered_map<const void*, SpCurrentCopy>>::iterator firstCopyMapWithNoCorrespondingHandlesIt,
                                   std::shared_ptr<std::atomic<size_t>>& numberOfSpeculativeSiblingSpecGroupsCounter,
                                   bool parentIsSpeculatif,
                                   const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params)
                                   -> decltype(coreTaskCreation(std::declval<std::array<CopyMapPtrTy, 1>>(),std::declval<SpTaskActivation>(),
                                   std::declval<const SpPriority&>(), std::forward_as_tuple(params...), std::make_index_sequence<sizeof...(ParamsAndTask)-1>{})){
                                       
        static_assert(SpecModel == SpSpeculativeModel::SP_MODEL_1
                      || SpecModel == SpSpeculativeModel::SP_MODEL_2
                      || SpecModel == SpSpeculativeModel::SP_MODEL_3, "Should not happen");

		auto tuple = std::forward_as_tuple(params...);
		auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        
        if constexpr (!isPotentialTask && allAreCopiableAndDeleteable<decltype (tuple)>(sequenceParamsNoFunction) == false){
            (void) firstCopyMapWithNoCorrespondingHandlesIt;
            for(; it != copyMaps.end(); it++){
                manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                removeAllCorrespondingCopies(*it, tuple, sequenceParamsNoFunction);
            }
            return coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                    SpTaskActivation::ENABLE, inPriority, std::move(tuple), sequenceParamsNoFunction);
        }else{
            static_assert(allAreCopiableAndDeleteable<decltype(tuple)>(sequenceParamsNoFunction) == true, "Add data passed to a potential task must be copiable");

            auto groups = getCorrespondingCopyGroups(*it, tuple, sequenceParamsNoFunction);
            bool oneGroupDisableOrFailed = false;
            
            for(auto gp : groups){
                if(gp->isSpeculationDisable() || gp->didSpeculationFailed() || gp->didParentSpeculationFailed()){
                    oneGroupDisableOrFailed = true;
                    break;
                }
            }
            
            const bool taskAlsoSpeculateOnOther = (groups.size() != 0 && !oneGroupDisableOrFailed);
            
            typename std::vector<std::unordered_map<const void*, SpCurrentCopy>>::iterator nextIt = std::next(it);
            
            if(nextIt != copyMaps.end() && !taskAlsoSpeculateOnOther) {
                manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                removeAllCorrespondingCopies(*it, tuple, sequenceParamsNoFunction);
                if(firstCopyMapWithNoCorrespondingHandlesIt == copyMaps.end()) {
                    firstCopyMapWithNoCorrespondingHandlesIt = it;
                }
                return preCoreTaskCreationAuxRec<isPotentialTask>(++it, firstCopyMapWithNoCorrespondingHandlesIt, numberOfSpeculativeSiblingSpecGroupsCounter, taskAlsoSpeculateOnOther || parentIsSpeculatif, inPriority, inProbability, params...);
            }else{
                if(taskAlsoSpeculateOnOther){
                    (*numberOfSpeculativeSiblingSpecGroupsCounter)++;
                }
                std::unique_ptr<SpGeneralSpecGroup<SpecModel>> currentGroupNormalTask = std::make_unique<SpGeneralSpecGroup<SpecModel>>(taskAlsoSpeculateOnOther || parentIsSpeculatif, numberOfSpeculativeSiblingSpecGroupsCounter);
                
                if constexpr(isPotentialTask) {
                    currentGroupNormalTask->setProbability(inProbability);
                }
                
                std::unique_ptr<SpGeneralSpecGroup<SpecModel>> finalSpecGroup(nullptr);
                
                std::unordered_map<const void*, SpCurrentCopy> l1p, l1, l2, l1l2, l1pModel3;
                
                if(taskAlsoSpeculateOnOther){
                    currentGroupNormalTask->addParents(groups);
                    if constexpr(isPotentialTask) {
                        currentSpecGroup = currentGroupNormalTask.get();
                        l1 = copyIfMaybeWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        assert(taskAlsoSpeculateOnOther == true || l1.size());
                        currentSpecGroup = nullptr;
                    }
                    
                    currentSpecGroup = currentGroupNormalTask.get();
                    l2 = copyIfWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                    currentSpecGroup = nullptr;
                    
                    l1l2.merge(l1);
                    l1l2.merge(l2);
                
                }else{
                    currentGroupNormalTask->setSpeculationActivation(true);
                    manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                    removeAllCorrespondingCopies(*it, tuple, sequenceParamsNoFunction);
                }
                
                if(nextIt == copyMaps.end() && taskAlsoSpeculateOnOther){
                    finalSpecGroup = std::make_unique<SpGeneralSpecGroup<SpecModel>>(taskAlsoSpeculateOnOther || parentIsSpeculatif, numberOfSpeculativeSiblingSpecGroupsCounter);
                    finalSpecGroup->setSpeculationActivation(true);
                }
                
                if constexpr(isPotentialTask) {
                    
                    currentSpecGroup = currentGroupNormalTask.get();
                    
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_1) {
                        if(taskAlsoSpeculateOnOther){
                            l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 2>{std::addressof(l1l2), std::addressof(*it)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        }else{
                            l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        }
                    }else if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_2){
                        if(finalSpecGroup != nullptr){
                            currentSpecGroup = finalSpecGroup.get();
                        }
                        l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                    }else if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(taskAlsoSpeculateOnOther){
                            l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 2>{std::addressof(l1l2), std::addressof(*it)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        }
                        if(nextIt == copyMaps.end()) {
                            if(finalSpecGroup != nullptr) {
                                currentSpecGroup = finalSpecGroup.get();
                            }
                            l1pModel3 = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, currentSpecGroup->getActivationStateForCopyTasks(), inPriority, tuple, sequenceParamsNoFunction);
                        }
                    }
                }
                
                currentSpecGroup = nullptr;
                
                using TaskViewTy = decltype(coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, currentGroupNormalTask->getActivationStateForMainTask(), inPriority, tuple, sequenceParamsNoFunction));
                TaskViewTy taskView, taskViewSpec;
                
                if(taskAlsoSpeculateOnOther){
                    taskViewSpec = coreTaskCreationSpeculative(std::array<CopyMapPtrTy, 2>{std::addressof(l1l2), std::addressof(*it)}, currentGroupNormalTask->getActivationStateForSpeculativeTask(), 
                                                                    inPriority, tuple, sequenceParamsNoFunction);
                    currentGroupNormalTask->setSpecTask(taskViewSpec.getTaskPtr());
                }
                
                if(nextIt == copyMaps.end()) {
                    if(finalSpecGroup != nullptr){
                        taskView = coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, finalSpecGroup->getActivationStateForMainTask(), inPriority, tuple, sequenceParamsNoFunction);
                    }else{
                        taskView = coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)}, currentGroupNormalTask->getActivationStateForMainTask(), inPriority, tuple, sequenceParamsNoFunction);
                    }
                }else {
                    size_t index = std::distance(copyMaps.begin(), it);
                    taskView = preCoreTaskCreationAuxRec<isPotentialTask>(++it, firstCopyMapWithNoCorrespondingHandlesIt, numberOfSpeculativeSiblingSpecGroupsCounter, taskAlsoSpeculateOnOther || parentIsSpeculatif, inPriority, inProbability, params...);
                    it = copyMaps.begin() + index;
                }
                
                currentGroupNormalTask->setMainTask(taskView.getTaskPtr());
                
                if(finalSpecGroup != nullptr) {
                    taskView.getTaskPtr()->setSpecGroup(finalSpecGroup.get());
                    finalSpecGroup->setMainTask(taskView.getTaskPtr());
                }else if(nextIt == copyMaps.end()) {
                    taskView.getTaskPtr()->setSpecGroup(currentGroupNormalTask.get());
                }
                
                if(!taskAlsoSpeculateOnOther) {
                    if constexpr(isPotentialTask) {
                        if(nextIt == copyMaps.end()) {
                            addCallbackToTask(taskView, currentGroupNormalTask.get());
                        }
                    }
                } else {
                    taskViewSpec.setOriginalTask(taskView.getTaskPtr());
                    
                    std::vector<SpAbstractTask*> mergeTasks = mergeIfInList(std::array<CopyMapPtrTy, 2>{std::addressof(l1l2), std::addressof(*it)}, currentGroupNormalTask.get(), inPriority, tuple, sequenceParamsNoFunction);
                    currentGroupNormalTask->addSelectTasks(mergeTasks);
                    manageReadDuplicate(*it, tuple, sequenceParamsNoFunction);
                    
                    if constexpr(isPotentialTask) {
                        addCallbackToTask(taskViewSpec, currentGroupNormalTask.get());
                    }
                    
                    if constexpr(isPotentialTask && SpecModel != SpSpeculativeModel::SP_MODEL_1) {
                        if(nextIt == copyMaps.end()) {
                            SpGeneralSpecGroup<SpecModel>* sg = currentGroupNormalTask.get();
                            if(finalSpecGroup != nullptr){
                                sg = finalSpecGroup.get();
                            }
                            addCallbackToTask(taskView, sg);
                        }
                    }
                }
                
                if constexpr(isPotentialTask) {
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(taskAlsoSpeculateOnOther) {
                            it->merge(l1p);
                        }
                    }else{
                         it->merge(l1p);
                    }
                
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(nextIt == copyMaps.end()) {
                            if(firstCopyMapWithNoCorrespondingHandlesIt == copyMaps.end() && (taskAlsoSpeculateOnOther || parentIsSpeculatif)) {
                                it = copyMaps.emplace(copyMaps.end());
                            }else if(firstCopyMapWithNoCorrespondingHandlesIt != copyMaps.end()){
                                it = firstCopyMapWithNoCorrespondingHandlesIt;
                            }
                            it->merge(l1pModel3);
                        }
                    }
                }
                
                specGroups.emplace_back(std::move(currentGroupNormalTask));
                
                if(finalSpecGroup != nullptr) {
                    specGroups.emplace_back(std::move(finalSpecGroup));
                }

                return taskView;
            }
        }
    }
    
    template <class... ParamsAndTask>
    inline auto preCoreTaskCreationSpec(const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params){
        return preCoreTaskCreationAux<true>(inPriority, inProbability, params...);
    }
    
    template <class... ParamsAndTask>
    inline auto preCoreTaskCreation(const SpPriority& inPriority, ParamsAndTask&&... params){
        return preCoreTaskCreationAux<false>(inPriority, SpProbability(1.0), params...);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <class Tuple, std::size_t IdxData, std::size_t N>
    std::vector<SpAbstractTask*> coreMergeIfInList(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                   SpGeneralSpecGroup<SpecModel> *sg,
                                                   const SpPriority& inPriority,
                                                   Tuple& args){
        using ScalarOrContainerType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        using TargetParamType = typename ScalarOrContainerType::RawHandleType;
        static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                      "They should all be default constructible here");
                      
        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;

        std::vector<SpAbstractTask*> mergeList;

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        
        
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            SpDataHandle* h1 = hh[indexHh];
            for(auto me : copyMapsToLookInto) {
                if(auto found = me->find(ptr); found != me->end()){
                    const SpCurrentCopy& cp = found->second;
                    assert(cp.isDefined());
                    assert(cp.originAdress == ptr);
                    assert(cp.latestAdress != nullptr);

                    assert(accessMode == SpDataAccessMode::READ || found->second.usedInRead == false);
                    
                    if(accessMode != SpDataAccessMode::READ){
                        bool isCarryingSurelyWrittenValuesOver = accessMode == SpDataAccessMode::WRITE;
                        
                        SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                        
                        auto taskView = this->taskInternalSpSelect(
                                           copyMapsToLookInto,
                                           isCarryingSurelyWrittenValuesOver,
                                           sg->getActivationStateForSelectTask(isCarryingSurelyWrittenValuesOver),
                                           inPriority,
                                           SpWrite(*h1->castPtr<TargetParamType>()),
                                           SpWrite(*h1copy->castPtr<TargetParamType>()),
                                           [](TargetParamType& output, TargetParamType& input){
                                               output = std::move(input);
                                           }
                        );
                        
                        taskView.setTaskName("sp-select");
                        mergeList.emplace_back(taskView.getTaskPtr());
                        
                        // delete copied data carried over by select task
                        taskInternal(copyMapsToLookInto, SpTaskActivation::ENABLE, SpPriority(0),
                                     SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                     [](TargetParamType& output){
                                         delete &output;
                                     }).setTaskName("sp-delete");
                        
                        // delete copy from copy map
                        me->erase(found);
                    }
                    else{
                       found->second.usedInRead = true;
                    }
                    
                    break;
                }
                
            }
            
            indexHh += 1;
        }
        
        return mergeList;
    }

    template <class Tuple, std::size_t... Is, std::size_t N>
    std::vector<SpAbstractTask*> mergeIfInList(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                               SpGeneralSpecGroup<SpecModel> *sg,
                                               const SpPriority& inPriority,
                                               Tuple& args,
                                               std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        // Add the handles
        std::vector<SpAbstractTask*> fullList;
        
        if constexpr(sizeof...(Is) > 0) {
            ([](std::vector<SpAbstractTask*> &l, std::vector<SpAbstractTask*>&& l2) {
                l.insert(l.end(), l2.begin(), l2.end());
            }(fullList, coreMergeIfInList<Tuple, Is>(copyMapsToLookInto, sg, inPriority, args)), ...);
        }
        
        return fullList;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    //! Copy an object and return this related info (the task is created and submited)
    template <class ObjectType>
    SpCurrentCopy coreCopyCreationCore(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto,
                                       const SpTaskActivation initialActivationState,
                                       const SpPriority& inPriority,
                                       ObjectType& objectToCopy) {
        using TargetParamType = typename std::remove_reference<ObjectType>::type;

        static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                      "Maybewrite data must be copiable");

        TargetParamType* ptr = new TargetParamType();
        const TargetParamType* originPtr = &objectToCopy;
        const TargetParamType* sourcePtr = originPtr;
        
        // Use the latest version of the data
        if(auto found = copyMapToLookInto.find(originPtr) ; found != copyMapToLookInto.end()){
            assert(found->second.latestAdress);
            sourcePtr = reinterpret_cast<TargetParamType*>(found->second.latestAdress);
        }

        SpDebugPrint() << "SpRuntime -- coreCopyCreationCore -- setup copy from " << sourcePtr << " to " << &ptr;
        SpAbstractTaskWithReturn<void>::SpTaskViewer taskView = taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMapToLookInto)},
                                                                            initialActivationState,
                                                                            inPriority,
                                                                            SpWrite(*ptr),
                                                                            SpRead(*sourcePtr),
                                                                            [](TargetParamType& output, const TargetParamType& input){
                SpDebugPrint() << "SpRuntime -- coreCopyCreationCore -- execute copy from " << &input << " to " << &output;
                output = input;
        });
        taskView.setTaskName("sp-copy");

        SpCurrentCopy cp;
        cp.latestAdress = ptr;
        cp.latestCopyTask = taskView.getTaskPtr();
        cp.lastestSpecGroup = currentSpecGroup;
        cp.originAdress = const_cast<TargetParamType*>(originPtr);
        cp.sourceAdress = const_cast<TargetParamType*>(sourcePtr);
        cp.deleter.reset(new SpDeleter<TargetParamType>());

        return cp;
    }

    //! Copy all the data of a mode if the access mode matches or if copyIfAlreadyDuplicate is true
    template <class Tuple, std::size_t IdxData, SpDataAccessMode targetMode, std::size_t N>
    std::vector<SpCurrentCopy> coreCopyIfAccess(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                const SpTaskActivation initialActivationState,
                                                const SpPriority& inPriority,
                                                const bool copyIfAlreadyDuplicate,
                                                const bool copyIfUsedInRead, Tuple& args){
        static_assert(N > 0, "coreCopyIfAccess -- You must provide at least one copy map for inspection.");
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;

        if constexpr (accessMode == targetMode){
            static_assert(std::is_default_constructible<TargetParamType>::value
                          && std::is_copy_assignable<TargetParamType>::value,
                          "Data must be copiable");

            std::vector<SpCurrentCopy> allCopies;

            auto& scalarOrContainerData = std::get<IdxData>(args);
            auto hh = getDataHandle(scalarOrContainerData);
            assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);

            long int indexHh = 0;
            for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                SpDataHandle* h1 = hh[indexHh];
                
                bool doCopy = false;
                std::unordered_map<const void*, SpCurrentCopy>* mPtr;
                
                if(copyIfAlreadyDuplicate) { // always copy regardless of the fact that the data might have been previously copied
                    doCopy = true;
                    for(auto me: copyMapsToLookInto) {
                        mPtr = me;
                        if(me->find(h1->castPtr<TargetParamType>()) != me->end()) {
                            break;
                        }
                    }
                }else if(copyIfUsedInRead){ // if data has already been previously copied then only copy if the previous copy is used in read
                    bool hasBeenFound = false;
                    for(auto me : copyMapsToLookInto) {
                        mPtr = me;
                        
                        if(auto found = me->find(h1->castPtr<TargetParamType>()); found != me->end()) {
                            hasBeenFound = true;
                            doCopy = found->second.usedInRead;
                            break;
                        }
                    }
                    if(!hasBeenFound) {
                        doCopy = true;
                    }
                }else{ // if none of the above has been triggered, copy the data only if it has not already been duplicated
                    doCopy = true;
                    for(auto me : copyMapsToLookInto) {
                        doCopy &= (me->find(h1->castPtr<TargetParamType>()) == me->end());
                    }
                    
                    mPtr = copyMapsToLookInto.back();
                }
                
                if(doCopy) {
                    SpCurrentCopy cp = coreCopyCreationCore(*mPtr, initialActivationState, inPriority, *h1->castPtr<TargetParamType>());
                    allCopies.push_back(cp);
                }

                indexHh += 1;
            }

            return allCopies;
        }
        else{
            (void) copyIfAlreadyDuplicate;
            (void) copyIfUsedInRead;
            (void) initialActivationState;
            (void) inPriority;
            return std::vector<SpCurrentCopy>();
        }
    }
    
    template <const bool copyIfAlreadyDuplicate, const bool copyIfUsedInRead, SpDataAccessMode targetMode, class Tuple, std::size_t... Is, std::size_t N>
    std::unordered_map<const void*, SpCurrentCopy> copyAux(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                           const SpTaskActivation initialActivationState,
                                                           const SpPriority& inPriority,
                                                           Tuple& args){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        
        std::unordered_map<const void*, SpCurrentCopy> copyMap;
        
        if constexpr(sizeof...(Is) > 0) {
            ([](std::unordered_map<const void*, SpCurrentCopy> &cm, std::vector<SpCurrentCopy>&& copies) {
                for(const SpCurrentCopy &c : copies) {
                   cm[c.originAdress] = c; 
                }
            }(copyMap, coreCopyIfAccess<Tuple, Is, targetMode>(copyMapsToLookInto, initialActivationState, inPriority, copyIfAlreadyDuplicate, copyIfUsedInRead, args)), ...);
        }
        
        return copyMap;
    }

    template <class Tuple, std::size_t... Is, std::size_t N>
    inline std::unordered_map<const void*, SpCurrentCopy> copyIfMaybeWriteAndNotDuplicateOrUsedInRead(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                                                                      const SpTaskActivation initialActivationState,
                                                                                                      const SpPriority& inPriority,
                                                                                                      Tuple& args,
                                                                                                      std::index_sequence<Is...>){
        return copyAux<false, true, SpDataAccessMode::MAYBE_WRITE, Tuple, Is...>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }

    template <class Tuple, std::size_t... Is, std::size_t N>
    inline std::unordered_map<const void*, SpCurrentCopy> copyIfMaybeWriteAndDuplicate(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                                                       const SpTaskActivation initialActivationState,
                                                                                       const SpPriority& inPriority,
                                                                                       Tuple& args, std::index_sequence<Is...>){
        return copyAux<true, false, SpDataAccessMode::MAYBE_WRITE, Tuple, Is...>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }

    template <class Tuple, std::size_t... Is, std::size_t N>
    inline std::unordered_map<const void*, SpCurrentCopy> copyIfWriteAndNotDuplicateOrUsedInRead(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                                                                 const SpTaskActivation initialActivationState,
                                                                                                 const SpPriority& inPriority,
                                                                                                 Tuple& args,
                                                                                                 std::index_sequence<Is...>){
        return copyAux<false, true, SpDataAccessMode::WRITE, Tuple, Is...>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <class Tuple, std::size_t IdxData>
    void coreManageReadDuplicate(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;

        if constexpr (std::is_destructible<TargetParamType>::value){
            auto hh = getDataHandle(scalarOrContainerData);
            assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
            long int indexHh = 0;
            for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                
                if(auto found = copyMap.find(ptr); found != copyMap.end()
                        && found->second.usedInRead
                        && accessMode != SpDataAccessMode::READ){
                    assert(std::is_copy_assignable<TargetParamType>::value);

                    SpCurrentCopy& cp = found->second;
                    assert(cp.latestAdress != ptr);
                    assert(cp.latestAdress);
                    this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMap)}, SpTaskActivation::ENABLE, SpPriority(0),
                                      SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                      [](TargetParamType& output){

                        delete &output;
                    }).setTaskName("sp-delete");

                    copyMap.erase(found);
                }

                indexHh += 1;
            }
            (void) hh;
        }
    }

    template <class Tuple, std::size_t... Is>
    void manageReadDuplicate(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        ((void) coreManageReadDuplicate<Tuple, Is>(copyMap, args), ...);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <class Tuple, std::size_t IdxData, class RetType>
    auto coreGetCorrespondingCopyAux(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        RetType res;

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

            if(auto found = copyMap.find(ptr); found != copyMap.end()){
                assert(found->second.lastestSpecGroup);
                if constexpr(SpUtils::is_instantiation_of<std::vector, RetType>::value) {
                    res.emplace_back(found->second.lastestSpecGroup);
                } else if constexpr(SpUtils::is_instantiation_of<std::unordered_map, RetType>::value){
                    res[ptr] = (found->second);
                }
            }

            indexHh += 1;
        }
        
        (void) hh;

        return res;
    }
    
    template <class Tuple, std::size_t IdxData>
    inline auto coreGetCorrespondingCopyGroups(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){
        return coreGetCorrespondingCopyAux<Tuple, IdxData, std::vector<SpGeneralSpecGroup<SpecModel>*>>(copyMap, args);
    }
    
    template <class Tuple, std::size_t... Is>
    std::vector<SpGeneralSpecGroup<SpecModel>*> getCorrespondingCopyGroups(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        
        std::vector<SpGeneralSpecGroup<SpecModel>*> result;
        
        // Add the handles
        if constexpr(sizeof...(Is) > 0) {
            ([](std::vector<SpGeneralSpecGroup<SpecModel>*> &res, std::vector<SpGeneralSpecGroup<SpecModel>*>&& cg) {
                res.insert(res.end(), cg.begin(), cg.end());
            }(result, coreGetCorrespondingCopyGroups<Tuple, Is>(copyMap, args)), ...);
        }
        
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <class Tuple, std::size_t IdxData>
    inline auto coreGetCorrespondingCopyList(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){
        return coreGetCorrespondingCopyAux<Tuple, IdxData, std::unordered_map<const void*, SpCurrentCopy> >(copyMap, args);
    }

    template <class Tuple, std::size_t... Is>
    std::unordered_map<const void*, SpCurrentCopy> getCorrespondingCopyList(std::unordered_map<const void*, SpCurrentCopy>& copyMap, 
                                                                            Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::unordered_map<const void*, SpCurrentCopy> result;
        
        // Add the handles
        if constexpr(sizeof...(Is) > 0) {
            ([](std::unordered_map<const void*, SpCurrentCopy> &res, std::unordered_map<const void*, SpCurrentCopy>&& group) {
                res.insert(group.begin(), group.end());
            }(result, coreGetCorrespondingCopyList<Tuple, Is>(copyMap, args)), ...);
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <class Tuple, std::size_t IdxData>
    void coreRemoveAllCorrespondingCopies(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto, Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;

        if constexpr (std::is_destructible<TargetParamType>::value){

            auto hh = getDataHandle(scalarOrContainerData);
            assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
            long int indexHh = 0;
            for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

                if(auto found = copyMapToLookInto.find(ptr); found != copyMapToLookInto.end()){
                    if(accessMode != SpDataAccessMode::READ){
                        assert(std::is_copy_assignable<TargetParamType>::value);
                        SpCurrentCopy& cp = found->second;
                        assert(cp.latestAdress);
                        this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMapToLookInto)}, SpTaskActivation::ENABLE, SpPriority(0),
                                          SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                          [ptr = cp.latestAdress](TargetParamType& output){
                                                assert(ptr ==  &output);
                                                delete &output;
                                          }).setTaskName("sp-delete");
                        copyMapToLookInto.erase(found);
                     }
                }

                indexHh += 1;
            }
            
            (void) hh;
        }
    }

    template <class Tuple, std::size_t... Is>
    void removeAllCorrespondingCopies(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        ((void) coreRemoveAllCorrespondingCopies<Tuple, Is>(copyMapToLookInto, args), ...);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////


    template <class Tuple, std::size_t IdxData>
    static constexpr bool coreAllAreCopiable(){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;

        return std::is_default_constructible<TargetParamType>::value
                && std::is_copy_assignable<TargetParamType>::value
                && std::is_destructible<TargetParamType>::value;
    }

    template <class Tuple, std::size_t... Is>
    static constexpr bool allAreCopiableAndDeleteable(std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        if constexpr(sizeof...(Is) > 0) {
            return (coreAllAreCopiable<Tuple, Is>() && ...);
        } else {
            return true;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <const bool isSpeculative, class Tuple, std::size_t IdxData, class TaskCorePtr, std::size_t N>
    void coreHandleCreationAux(Tuple& args, TaskCorePtr& aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
        using ScalarOrContainerType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;
        static_assert(!isSpeculative || (std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value),
                      "They should all be default constructible here");

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            SpDataHandle* h = hh[indexHh];
            
            bool foundAddressInCopies = false;
            void *cpLatestAddress = nullptr;
            
            if constexpr(isSpeculative){
                for(auto me : copyMapsToLookInto) {
                    if(auto found = me->find(ptr); found != me->end()){
                        const SpCurrentCopy& cp = found->second;
                        assert(cp.isDefined());
                        assert(cp.originAdress == ptr);
                        assert(cp.latestAdress != nullptr);
                        h = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                        cpLatestAddress = cp.latestAdress;
                        foundAddressInCopies = true;
                        break;
                    }
                }
            }
            
            if(!foundAddressInCopies) {
                SpDebugPrint() << "accessMode in runtime to add dependence -- => " << SpModeToStr(accessMode);
            }
            
            const long int handleKey = h->addDependence(aTask, accessMode);
            if(indexHh == 0){
                aTask->template setDataHandle<IdxData>(h, handleKey);
            }
            else{
                assert(ScalarOrContainerType::IsScalar == false);
                aTask->template addDataHandleExtra<IdxData>(h, handleKey);
            }
            
            if constexpr(isSpeculative) {
                if(foundAddressInCopies) {
                    aTask->template updatePtr<IdxData>(indexHh, reinterpret_cast<TargetParamType*>(cpLatestAddress));
                }
            }
            
            (void) cpLatestAddress;
            (void) ptr;
            
            indexHh += 1;
        }
    }
    
    template <class Tuple, std::size_t IdxData, class TaskCorePtr, std::size_t N>
    inline void coreHandleCreationSpec(Tuple& args, TaskCorePtr& aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
       coreHandleCreationAux<true, Tuple, IdxData, TaskCorePtr>(args, aTask, copyMapsToLookInto);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////
    
    template <template<typename...> typename TaskType, class... ParamsAndTask, std::size_t N>
    auto taskInternal(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                            const SpTaskActivation inActivation,
                            const SpPriority &inPriority,
                            ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        return coreTaskCreationAux<TaskType, false>(inActivation, inPriority, std::forward_as_tuple(inParamsAndTask...), sequenceParamsNoFunction, copyMapsToLookInto);
    }
    
    
    template <class... ParamsAndTask, std::size_t N>
    inline auto taskInternal(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                             const SpTaskActivation inActivation,
                             const SpPriority &inPriority,
                             ParamsAndTask&&... inParamsAndTask){
        return taskInternal<SpTask>(copyMapsToLookInto, inActivation, inPriority, inParamsAndTask...);
    }
    
    template <class... ParamsAndTask, std::size_t N>
    auto taskInternalSpSelect(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                              bool isCarryingSurelyWrittenValuesOver,
                              const SpTaskActivation inActivation,
                              const SpPriority &inPriority,
                              ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        return coreTaskCreationAux<SpSelectTask, false>(inActivation, inPriority, std::forward_as_tuple(inParamsAndTask...), sequenceParamsNoFunction, 
                                                        copyMapsToLookInto, isCarryingSurelyWrittenValuesOver);
    }


    ///////////////////////////////////////////////////////////////////////////
    std::function<bool(int,const SpProbability&)> specFormula;

public:
    void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula){
        specFormula = std::move(inFormula);
    }

    void thisTaskIsReady(SpAbstractTask* aTask) final {
        SpGeneralSpecGroup<SpecModel>* specGroup = aTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
        SpDebugPrint() << "SpRuntime -- thisTaskIsReady -- will test ";
        if(specGroup && specGroup->isSpeculationNotSet()){
            if(specGroup != currentSpecGroup || SpUtils::GetThreadId() != 0){
                specGroupMutex.lock();
            }
            if(specGroup->isSpeculationNotSet()){
                if(specFormula){
                    if(specFormula(scheduler.getNbReadyTasks(), specGroup->getAllProbability())){
                        SpDebugPrint() << "SpRuntime -- thisTaskIsReady -- enableSpeculation ";
                        specGroup->setSpeculationActivation(true);
                    }
                    else{
                        specGroup->setSpeculationActivation(false);
                    }
                }
                else if(scheduler.getNbReadyTasks() == 0){
                    SpDebugPrint() << "SpRuntime -- thisTaskIsReady -- enableSpeculation ";
                    specGroup->setSpeculationActivation(true);
                }
                else{
                    specGroup->setSpeculationActivation(false);
                }
            }
            if(specGroup != currentSpecGroup || SpUtils::GetThreadId() != 0){
                specGroupMutex.unlock();
            }
            assert(!specGroup->isSpeculationNotSet());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////
    /// Constructor
    ///////////////////////////////////////////////////////////////////////////

    explicit SpRuntime(const int inNumThreads = SpUtils::DefaultNumThreads())
        : nbThreads(inNumThreads), currentSpecGroup(nullptr){
        threads.reserve(static_cast<unsigned long>(inNumThreads));
        // Bind only if enough core
        const bool bindToCore = (inNumThreads <= SpUtils::DefaultNumThreads());

        // Create all threads
        for(int idxThread = 0 ; idxThread < inNumThreads ; ++idxThread){
            threads.emplace_back([this, idxThread, bindToCore](){
                // Set id
                SpUtils::SetThreadId(idxThread+1);
                // Bind to core
                if(bindToCore){
                    SpUtils::BindToCore(idxThread);
                }

                SpDebugPrint() << "Starts";
                // Call scheduler
                scheduler.runnerCallback();
                SpDebugPrint() << "Ends";
            });
        }

        scheduler.registerListener(this);
    }

    // No copy and no move
    SpRuntime(const SpRuntime&) = delete;
    SpRuntime(SpRuntime&&) = delete;
    SpRuntime& operator=(const SpRuntime&) = delete;
    SpRuntime& operator=(SpRuntime&&) = delete;

    //! Destructor waits tasks and stop threads
    ~SpRuntime(){
        waitAllTasks();
        stopAllThreads();
        // free copies
        releaseCopies();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Getters/actions
    ///////////////////////////////////////////////////////////////////////////

    int getNbThreads() const{
        return nbThreads;
    }

    void stopAllThreads(){
        scheduler.stopAllWorkers();
        for(auto&& thread : threads){
            thread.join();
        }
        threads.clear();
    }

    void waitAllTasks(){
        scheduler.waitAllTasks();
    }

    void waitRemain(const long int windowSize){
        scheduler.waitRemain(windowSize);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Tasks creation methods
    ///////////////////////////////////////////////////////////////////////////

    template <class... ParamsAndTask>
    auto task(SpPriority inPriority, ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        CheckPrototypeCore<decltype(std::forward_as_tuple(inParamsAndTask...))>(sequenceParamsNoFunction);
        return preCoreTaskCreation<ParamsAndTask...>(inPriority, std::forward<ParamsAndTask>(inParamsAndTask)...);
    }

    template <class... ParamsAndTask>
    auto task(ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        CheckPrototypeCore<decltype(std::forward_as_tuple(inParamsAndTask...))>(sequenceParamsNoFunction);
        return preCoreTaskCreation<ParamsAndTask...>(SpPriority(0), std::forward<ParamsAndTask>(inParamsAndTask)...);
    }

    template <class... ParamsAndTask>
    auto potentialTask(SpPriority inPriority, SpProbability inProbability, ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        CheckPrototypeCore<decltype(std::forward_as_tuple(inParamsAndTask...))>(sequenceParamsNoFunction);
        return preCoreTaskCreationSpec<ParamsAndTask...>(inPriority, inProbability, std::forward<ParamsAndTask>(inParamsAndTask)...);
    }

    template <class... ParamsAndTask>
    auto potentialTask(SpProbability inProbability, ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        CheckPrototypeCore<decltype(std::forward_as_tuple(inParamsAndTask...))>(sequenceParamsNoFunction);
        return preCoreTaskCreationSpec<ParamsAndTask...>(SpPriority(0), inProbability, std::forward<ParamsAndTask>(inParamsAndTask)...);
    }

    template <class... ParamsAndTask>
    auto potentialTask(ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        CheckPrototypeCore<decltype(std::forward_as_tuple(inParamsAndTask...))>(sequenceParamsNoFunction);
        return preCoreTaskCreationSpec<ParamsAndTask...>(SpPriority(0), SpProbability(), std::forward<ParamsAndTask>(inParamsAndTask)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Output
    ///////////////////////////////////////////////////////////////////////////

    void generateDot(const std::string& outputFilename) const {
        SpDotDag::GenerateDot(outputFilename, scheduler.getFinishedTaskList());
    }


    void generateTrace(const std::string& outputFilename, const bool showDependences = true) const {
        SpSvgTrace::GenerateTrace(outputFilename, scheduler.getFinishedTaskList(), startingTime, getNbThreads(), showDependences);

    }
};


#endif
