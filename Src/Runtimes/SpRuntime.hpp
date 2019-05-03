///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPRUNTIME_HPP
#define SPRUNTIME_HPP

#include <functional>
#include <list>
#include <unordered_map>
#include <memory>
#include <unistd.h>
#include <cmath>
#include <type_traits>

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


//! The runtime is the main component of spetabaru.
class SpRuntime : public SpAbstractToKnowReady {
    template <class Tuple, std::size_t IdxData>
    static bool CheckParam(){
        using ParamType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        static_assert(has_getView<ParamType>::value, "Converted object to a task must have getView method");
        static_assert(has_getAllData<ParamType>::value, "Converted object to a task must have getAllData method");
        return true;
    }

    template <class Tuple, std::size_t... Is>
    static auto CheckPrototypeCore(std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value >= 1, "You must pass at least a function to create a task");

        bool do_in_order[] = {0, CheckParam<Tuple, Is>() ...};
        (void)do_in_order;

        using TaskCore = typename std::remove_reference<typename std::tuple_element<std::tuple_size<Tuple>::value-1, Tuple>::type>::type;
        static_assert(std::is_invocable<TaskCore, decltype(std::declval<typename std::tuple_element<Is, Tuple>::type>().getView()) ...>::value,
                      "Impossible to invoc le last argument passing all the previous ones");
    }

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

    template <std::size_t IdxData, class ParamsType, class TaskCorePtr>
    typename std::enable_if<ParamsType::IsScalar,bool>::type
    createHandleAndAddToTaskCore(ParamsType& scalarData, TaskCorePtr& aTask){
        SpDataHandle* h1 = getDataHandle(scalarData)[0];
        const SpDataAccessMode accessMode = ParamsType::AccessMode;
        SpDebugPrint() << "accessMode in runtime to add dependence -- => " << SpModeToStr(accessMode);
        const long int handleKey1 = h1->addDependence(aTask, accessMode);
        aTask->template setDataHandle<IdxData>(h1, handleKey1);
        return true;
    }

    template <std::size_t IdxData, class ParamsType, class TaskCorePtr>
    typename std::enable_if<!ParamsType::IsScalar,bool>::type
    createHandleAndAddToTaskCore(ParamsType& containerData, TaskCorePtr& aTask){
        std::vector<SpDataHandle*> hh = getDataHandle(containerData);
        bool isFirstHandle = true;
        for(SpDataHandle* h1 : hh){
            const long int handleKey1 = h1->addDependence(aTask, ParamsType::AccessMode);
            if(isFirstHandle){
                aTask->template setDataHandle<IdxData>(h1, handleKey1);
                isFirstHandle = false;
            }
            else{
                aTask->template addDataHandleExtra<IdxData>(h1, handleKey1);
            }
        }
        return true;
    }

    template <class Tuple, std::size_t IdxData, class TaskCorePtr>
    bool createHandleAndAddToTask(Tuple& args, TaskCorePtr& aTask){
        auto& scalarOrContainerData = std::get<IdxData>(args);
        return createHandleAndAddToTaskCore<IdxData>(scalarOrContainerData, aTask);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Core part of the task creation
    ///////////////////////////////////////////////////////////////////////////

    //! Convert tuple to data and call the function
    //! Args is a value to allow for move or pass a rvalue reference
    template <class Tuple, std::size_t... Is>
    auto coreTaskCreation(const SpTaskActivation inActivation, const SpPriority& inPriority, Tuple args, std::index_sequence<Is...>){ // (?,priorité, tuple de ref ?, données de la tâche ?)
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        SpDebugPrint() << "SpRuntime -- coreTaskCreation";

        // Get the type of the function (maybe class, lambda, etc.)
        using TaskCore = typename std::remove_reference<typename std::tuple_element<std::tuple_size<Tuple>::value-1, Tuple>::type>::type; //fonction est la dernière valeur du tuple ?

        // Get the task object
        TaskCore taskCore = std::move(std::get<(std::tuple_size<Tuple>::value-1)>(args)); // dernière valeur du tuple ? 

        // Get the return type of the task (can be void)
        using RetType = decltype(taskCore(std::get<Is>(args).getView()...));

        // Create a task with a copy of the args
        auto aTask = new SpTask<TaskCore, RetType,
                typename std::remove_reference_t<typename std::tuple_element<Is, Tuple>::type> ... >(std::move(taskCore), inPriority,
                                                                   std::make_tuple(std::get<Is>(args)...)); //type du tuple {tache,données} ?

        // Lock the task
        aTask->takeControl();

        // Add the handles
        bool do_in_order[] = {0, createHandleAndAddToTask<Tuple, Is, decltype(aTask)>(args, aTask) ...}; //ordonnancement des dépendances ? bool ? 
        (void)do_in_order; 

        // Check coherency
        assert(aTask->areDepsCorrect());

        // The task has been initialized
        aTask->setState(SpTaskState::INITIALIZED);
        aTask->setEnabled(inActivation);

        // Get the view
        typename SpAbstractTaskWithReturn<RetType>::SpTaskViewer descriptor = aTask->getViewer(); //view ?

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

    ///////////////////////////////////////////////////////////////////////////
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

        void deleteObject(void* ptr) override{
            delete reinterpret_cast<ObjectType*>(ptr);
        }
    };

    struct SpCurrentCopy{
        SpCurrentCopy()
            :originAdress(nullptr), sourceAdress(nullptr), latestAdress(nullptr), lastestSpecGroup(nullptr),
              latestCopyTask(nullptr), usedInRead(false), hasBeenDeleted(false){
        }

        bool isDefined() const{
            return originAdress != nullptr;
        }

        void* originAdress;
        void* sourceAdress;
        void* latestAdress;
        SpGeneralSpecGroup* lastestSpecGroup;
        SpAbstractTask* latestCopyTask;
        bool usedInRead;
        bool hasBeenDeleted;

        std::shared_ptr<SpAbstractDeleter> deleter;
    };

    std::unordered_map<const void*,SpCurrentCopy> copiedHandles;
    SpGeneralSpecGroup* currentSpecGroup;
    std::list<std::unique_ptr<SpGeneralSpecGroup>> specGroups;
    std::mutex specGroupMutex;

    void releaseCopies(){
        for(auto& iter : copiedHandles){
            assert(iter.second.hasBeenDeleted == false);
            assert(iter.second.latestAdress);
            assert(iter.second.deleter);
            iter.second.deleter->deleteObject(iter.second.latestAdress);
        }
        copiedHandles.clear();
    }

    //! Convert tuple to data and call the function
    template <class Tuple, std::size_t... Is>
    auto coreTaskCreationSpeculative(std::unordered_map<const void*, SpCurrentCopy>& extraCopies, const SpTaskActivation inActivation,
                                     const SpPriority& inPriority, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        SpDebugPrint() << "SpRuntime -- coreTaskCreation";

        // Get the type of the function (maybe class, lambda, etc.)
        using TaskCore = typename std::remove_reference<typename std::tuple_element<std::tuple_size<Tuple>::value-1, Tuple>::type>::type;

        // Get the task object
        TaskCore taskCore = std::move(std::get<(std::tuple_size<Tuple>::value-1)>(args));

        // Get the return type of the task (can be void)
        using RetType = decltype(taskCore(GetRealData(std::get<Is>(args)) ...));

        // Create a task with a copy of the args
        auto aTask = new SpTask<TaskCore, RetType,
                typename std::remove_reference_t<typename std::tuple_element<Is, Tuple>::type> ... >(std::move(taskCore), inPriority,
                                                                                                     std::make_tuple(std::get<Is>(args)...));

        // Lock the task
        aTask->takeControl();

        // Add the handles
        bool do_in_order[] = {0, coreHandleCreationSpec<Tuple, Is, decltype(aTask)>(args, aTask, extraCopies) ...};
        (void)do_in_order;

        // Check coherency
        assert(aTask->areDepsCorrect());

        // The task has been initialized
        aTask->setState(SpTaskState::INITIALIZED);
        aTask->setEnabled(inActivation);

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


    //! Create tuple and indexes to simplify work in coreTaskCreation
    template <class... ParamsAndTask>
    auto preCoreTaskCreation(const SpPriority& inPriority, ParamsAndTask&&... params){ //priorité, paramètre(s) de la tâche et la tâche ?
        auto tuple = std::forward_as_tuple(params...); //tuple de references vers params ...
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{}; //sequence d'entier à partir du nombre de paramètres du template ?

        manageReadDuplicate(tuple, sequenceParamsNoFunction); //
    
        if constexpr (allAreCopiableAndDeleteable<decltype (tuple)>(sequenceParamsNoFunction) == false){ //si toutes les données ne sont pas copiable et deletable ?
            removeAllCorrespondingCopies(tuple, sequenceParamsNoFunction);
            return coreTaskCreation(SpTaskActivation::ENABLE, inPriority, std::move(tuple), sequenceParamsNoFunction);
        }
        else{ // pourquoi les groupes sont uniquement utilisables lorsque toutes les données sont copiable/deletable ? groupes nécessaire pour mon modèle ?
            scheduler.lockListenersReadyMutex();
            specGroupMutex.lock();

            std::vector<SpGeneralSpecGroup*> groups = getCorrespondingCopyGroups(tuple, sequenceParamsNoFunction);

            bool oneGroupDisableOrFailed = false;
            for(SpGeneralSpecGroup* gp : groups){
                if(gp->isSpeculationDisable() || gp->didSpeculationFailed() || gp->didParentSpeculationFailed()){
                    oneGroupDisableOrFailed = true;
                    break;
                }
            }

            if(groups.size() == 0 || oneGroupDisableOrFailed){
                removeAllCorrespondingCopies(tuple, sequenceParamsNoFunction);
                specGroupMutex.unlock();
                scheduler.unlockListenersReadyMutex();
                return coreTaskCreation(SpTaskActivation::ENABLE, inPriority, std::move(tuple), sequenceParamsNoFunction);
            }

            std::unique_ptr<SpGeneralSpecGroup> currentGroupNormalTask(new SpGeneralSpecGroup(true));
            currentGroupNormalTask->addParents(groups);

            std::unordered_map<const void*, SpCurrentCopy> l2;

            currentSpecGroup = currentGroupNormalTask.get();
            l2 = copyIfWriteAndNotDuplicate(inPriority, tuple, sequenceParamsNoFunction); // une autre donée est utilisé mais pas dupliqué (pas dans notre cas on utilise une seule donée)
            currentSpecGroup = nullptr;

            auto taskView = coreTaskCreation(SpTaskActivation::ENABLE, inPriority, tuple, sequenceParamsNoFunction);

            currentGroupNormalTask->setMainTask(taskView.getTaskPtr());

            currentGroupNormalTask->addCopyTasks(copyMapToTaskVec(l2));

            auto taskViewSpec = coreTaskCreationSpeculative(l2, SpTaskActivation::ENABLE, inPriority, tuple, sequenceParamsNoFunction);
            taskViewSpec.setOriginalTask(taskView.getTaskPtr());

            currentGroupNormalTask->setSpecTask(taskViewSpec.getTaskPtr());

            std::vector<SpAbstractTask*> mergeTasks = mergeIfInList(l2, inPriority, tuple, sequenceParamsNoFunction); //select

            currentGroupNormalTask->addSelectTasks(std::move(mergeTasks));

            removeAllCorrespondingCopies(tuple, sequenceParamsNoFunction);

            specGroups.emplace_back(std::move(currentGroupNormalTask));

            specGroupMutex.unlock();
            scheduler.unlockListenersReadyMutex();

            return taskView;
        }
    }


    static std::vector<SpAbstractTask*> copyMapToTaskVec(const std::unordered_map<const void*, SpCurrentCopy>& map){
        std::vector<SpAbstractTask*> list;
        list.reserve(map.size());
        for( auto const& cp : map ) {
            list.emplace_back(cp.second.latestCopyTask);
        }
        return list;
    }

    //! Create tuple and indexes to simplify work in coreTaskCreation
    template <class... ParamsAndTask>
    auto preCoreTaskCreationSpec(const SpPriority& inPriority, const SpProbability& inProbability, ParamsAndTask&&... params){
      
        auto tuple = std::forward_as_tuple(params...); //tuple de références vers params ...
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{}; // construit une suite d'index en se basant sur le nombre de param -1 ?

        static_assert(allAreCopiableAndDeleteable<decltype(tuple)>(sequenceParamsNoFunction) == true,
                      "Add data passed to a potential task must be copiable"); // templater allAreCopiableAndDeleteable avec le type déduit par decltype, simple assert

        manageReadDuplicate(tuple, sequenceParamsNoFunction);

        scheduler.lockListenersReadyMutex();
        specGroupMutex.lock();

        std::vector<SpGeneralSpecGroup*> groups = getCorrespondingCopyGroups(tuple, sequenceParamsNoFunction); 
        bool oneGroupDisableOrFailed = false;
        for(SpGeneralSpecGroup* gp : groups){
            if(gp->isSpeculationDisable() || gp->didSpeculationFailed() || gp->didParentSpeculationFailed()){
                oneGroupDisableOrFailed = true;
                break;
            }
        }
        const bool taskAlsoSpeculateOnOther = (groups.size() != 0 && !oneGroupDisableOrFailed);
              
        std::unique_ptr<SpGeneralSpecGroup> currentGroupNormalTask(new SpGeneralSpecGroup(taskAlsoSpeculateOnOther));
        currentGroupNormalTask->setProbability(inProbability);
        if(taskAlsoSpeculateOnOther == true){ 
            currentGroupNormalTask->addParents(groups);
        }

        std::unordered_map<const void*, SpCurrentCopy> copiesBeforeCurrentTask = getCorrespondingCopyList(tuple, sequenceParamsNoFunction);
        
        for(auto& iter : copiesBeforeCurrentTask){
            assert(copiedHandles.find(iter.first) != copiedHandles.end());
            copiedHandles.erase(iter.first);
        }

        //mapping données
        std::unordered_map<const void*, SpCurrentCopy> l1;

                          
        //@debug
        std::cout<<"group size: "<<groups.size()<<"\n"; 
        std::cout<<"!oneGroupDisableOrFailed: "<<!oneGroupDisableOrFailed<<"\n";
        std::cout<<"taskAlsoSpeculateOnOther: "<<taskAlsoSpeculateOnOther<<"\n";
                
        //nécessaire
        currentSpecGroup = currentGroupNormalTask.get();
        // Always create copy tasks
        l1 = copyIfMaybeWriteAndDuplicate(inPriority, tuple, sequenceParamsNoFunction);
        currentSpecGroup = nullptr;
        currentGroupNormalTask->addCopyTasks(copyMapToTaskVec(l1));           
        
        
        // Create and insert normal task on the regular data
        auto taskView = coreTaskCreation(SpTaskActivation::ENABLE, inPriority, tuple, sequenceParamsNoFunction);
        // C)
        currentGroupNormalTask->setMainTask(taskView.getTaskPtr());
          
        if(taskAlsoSpeculateOnOther == false){
            // removeAllCorrespondingCopies(tuple, sequenceParamsNoFunction);
            for(auto& cp : l1){ //nécessaire pour la création du graphe. Pourquoi cela était fait après l'ajout du callback sur la tache ? 
                assert(copiedHandles.find(cp.first) == copiedHandles.end());
                copiedHandles[cp.first] = cp.second;
                copiedHandles[cp.first].lastestSpecGroup = currentGroupNormalTask.get();
            }
            
        } else {
            //créer et insérer t' sur la copie A)
            auto taskViewSpec = coreTaskCreationSpeculative(copiesBeforeCurrentTask, SpTaskActivation::ENABLE, inPriority, tuple, sequenceParamsNoFunction); // l1l2 vide
            taskViewSpec.setOriginalTask(taskView.getTaskPtr());            
            currentGroupNormalTask->setSpecTask(taskViewSpec.getTaskPtr());

            // D) création select
            std::vector<SpAbstractTask*> mergeTasks = mergeIfInList(copiesBeforeCurrentTask, inPriority, tuple, sequenceParamsNoFunction);
            currentGroupNormalTask->addSelectTasks(std::move(mergeTasks));

            removeAllCorrespondingCopiesList(copiesBeforeCurrentTask, tuple, sequenceParamsNoFunction);
          
            // E) mettre a jour la liste gloable des copies avec la liste issue de B)
            for(auto& cp : l1){
                assert(copiedHandles.find(cp.first) == copiedHandles.end());
                copiedHandles[cp.first] = cp.second;
                copiedHandles[cp.first].lastestSpecGroup = currentGroupNormalTask.get();
            }
                        
            
            //add callback to spec task            
          taskViewSpec.addCallback([this, aTaskPtr = taskViewSpec.getTaskPtr(), specGroupPtr = currentGroupNormalTask.get()]
                             (const bool alreadyDone, const bool& taskRes, SpAbstractTaskWithReturn<bool>::SpTaskViewer& /*view*/,
                             const bool isEnabled) {
                if(!alreadyDone) {
                        assert(SpUtils::GetThreadId() != 0);
                        specGroupMutex.lock();
                }

                specGroupPtr->setSpeculationCurrentResult(!taskRes, isEnabled, false);

                if(!alreadyDone) {
                        assert(SpUtils::GetThreadId() != 0);
                        specGroupMutex.unlock();
                }
            });
            
        }
        
        assert(taskAlsoSpeculateOnOther == true || l1.size());

        taskView.addCallback([this, aTaskPtr = taskView.getTaskPtr(), specGroupPtr = currentGroupNormalTask.get()]
                             (const bool alreadyDone, const bool& taskRes, SpAbstractTaskWithReturn<bool>::SpTaskViewer& /*view*/,
                             const bool isEnabled) {
              if(!alreadyDone) {
                    assert(SpUtils::GetThreadId() != 0);
                    specGroupMutex.lock();
              }

               specGroupPtr->setSpeculationCurrentResult(!taskRes, isEnabled, true);

              if(!alreadyDone){
                      assert(SpUtils::GetThreadId() != 0);
                      specGroupMutex.unlock();
              }
        });

        specGroups.emplace_back(std::move(currentGroupNormalTask));

        specGroupMutex.unlock();
        scheduler.unlockListenersReadyMutex();

        return taskView;
    }
    


    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t IdxData>
    std::vector<SpAbstractTask*> coreMergeIfInList(const SpPriority& inPriority,
                                                   Tuple& args, std::unordered_map<const void*, SpCurrentCopy>& extraCopies){
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

            if(auto found = extraCopies.find(ptr); found != extraCopies.end()){
                assert(copiedHandles.find(ptr) == copiedHandles.end());
                const SpCurrentCopy& cp = found->second;
                assert(cp.isDefined());
                assert(cp.originAdress == ptr);
                assert(cp.latestAdress != nullptr);

                assert(accessMode == SpDataAccessMode::READ || found->second.usedInRead == false);
                if(accessMode != SpDataAccessMode::READ){
                    SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                    auto taskView = this->taskInternal(SpTaskActivation::DISABLE, inPriority,
                                       SpWrite(*h1->castPtr<TargetParamType>()),
                                       SpWrite(*h1copy->castPtr<TargetParamType>()),
                                       [](TargetParamType& output, TargetParamType& input){
                        output = std::move(input);
                        delete &input;
                    });
                    taskView.setTaskName("sp-merge");
                    mergeList.emplace_back(taskView.getTaskPtr());
                    found->second.hasBeenDeleted = true;
                    found->second.latestAdress = nullptr;
                }
                else{
                   found->second.usedInRead = true;
                }
            }
            else if(auto found2 = copiedHandles.find(ptr); found2 != copiedHandles.end()){
                const SpCurrentCopy& cp = found2->second;
                assert(cp.isDefined());
                assert(cp.originAdress == ptr);
                assert(cp.latestAdress != nullptr);

                assert(accessMode == SpDataAccessMode::READ || found2->second.usedInRead == false);
                if(accessMode != SpDataAccessMode::READ){
                    SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                    auto taskView = this->taskInternal(SpTaskActivation::DISABLE, inPriority,
                                       SpWrite(*h1->castPtr<TargetParamType>()),
                                       SpWrite(*h1copy->castPtr<TargetParamType>()),
                                       [](TargetParamType& output, TargetParamType& input){
                        output = std::move(input);
                        delete &input;
                    });
                    taskView.setTaskName("sp-merge");
                    mergeList.emplace_back(taskView.getTaskPtr());
                    found2->second.hasBeenDeleted = true;
                    found2->second.latestAdress = nullptr;
                }
                else{
                   found2->second.usedInRead = true;
                }
            }

            indexHh += 1;
        }
        return mergeList;
    }

    template <class Tuple, std::size_t... Is>
    std::vector<SpAbstractTask*> mergeIfInList(std::unordered_map<const void*, SpCurrentCopy>& extraCopie,
                                               const SpPriority& inPriority,
                                               Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        // Add the handles
        std::vector<SpAbstractTask*> merged[] = {std::vector<SpAbstractTask*>(), coreMergeIfInList<Tuple, Is>(inPriority, args, extraCopie) ...};
        constexpr int nbLists = sizeof(merged)/sizeof(std::vector<SpAbstractTask*>);
        static_assert(sizeof...(Is) == nbLists-1, "oups");
        std::vector<SpAbstractTask*> fullList;
        for(int idxList = 0 ; idxList < nbLists ; ++idxList){
            fullList.insert(fullList.end(), merged[idxList].begin(), merged[idxList].end());
        }
        return fullList;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    //! Copy an object and return this related info (the task is created and submited)
    template <class ObjectType>
    SpCurrentCopy coreCopyCreationCore(const SpPriority& inPriority, ObjectType& objectToCopy) {
        using TargetParamType = typename std::remove_reference<ObjectType>::type;

        static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                      "Maybewrite data must be copiable");

        TargetParamType* ptr = new TargetParamType();
        const TargetParamType* originPtr = &objectToCopy;
        const TargetParamType* sourcePtr = originPtr;

        // Use the latest version of the data
        if(auto found = copiedHandles.find(originPtr) ; found != copiedHandles.end()){
            assert(found->second.latestAdress);
            sourcePtr = reinterpret_cast<TargetParamType*>(found->second.latestAdress);
        }

        SpDebugPrint() << "SpRuntime -- coreCopyCreationCore -- setup copy from " << sourcePtr << " to " << &ptr;
        SpAbstractTaskWithReturn<void>::SpTaskViewer taskView = this->taskInternal(SpTaskActivation::DISABLE, inPriority,
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
        cp.originAdress = const_cast<TargetParamType*>(originPtr);
        cp.sourceAdress = const_cast<TargetParamType*>(sourcePtr);
        cp.deleter.reset(new SpDeleter<TargetParamType>());

        return cp;
    }

    //! Copy all the data of a mode if the access mode matches or if copyIfAlreadyDuplicate is true
    template <class Tuple, std::size_t IdxData, SpDataAccessMode targetMode>
    std::vector<SpCurrentCopy> coreCopyIfAccess(const SpPriority& inPriority,
                                                const bool copyIfAlreadyDuplicate, Tuple& args){
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

                if(copyIfAlreadyDuplicate == true
                        || copiedHandles.find(h1->castPtr<TargetParamType>()) == copiedHandles.end()){
                    SpCurrentCopy cp = coreCopyCreationCore(inPriority, *h1->castPtr<TargetParamType>());
                    allCopies.push_back(cp);
                }

                indexHh += 1;
            }

            return allCopies;
        }
        else{
            (void)inPriority;
            return std::vector<SpCurrentCopy>();
        }
    }

    template <class Tuple, std::size_t... Is>
    std::unordered_map<const void*, SpCurrentCopy> copyIfMaybeWriteAndNotDuplicate(const SpPriority& inPriority, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::unordered_map<const void*, SpCurrentCopy> result;
        // Add the handles
        std::vector<SpCurrentCopy> copies[] = {std::vector<SpCurrentCopy>(), coreCopyIfAccess<Tuple, Is, SpDataAccessMode::MAYBE_WRITE>(inPriority, false, args) ...};
        constexpr int nbCopies = sizeof(copies)/sizeof(std::vector<SpCurrentCopy>);
        static_assert(sizeof...(Is) == nbCopies-1, "oups");
        for(int idxCopy = 0 ; idxCopy < nbCopies ; ++idxCopy){
            for(const SpCurrentCopy& cp : copies[idxCopy]){
                result[cp.originAdress] = cp;
            }
        }
        return result;
    }

    template <class Tuple, std::size_t... Is>
    std::unordered_map<const void*, SpCurrentCopy> copyIfMaybeWriteAndDuplicate(const SpPriority& inPriority, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::unordered_map<const void*, SpCurrentCopy> result;
        // Add the handles
        std::vector<SpCurrentCopy> copies[] = {std::vector<SpCurrentCopy>(), coreCopyIfAccess<Tuple, Is, SpDataAccessMode::MAYBE_WRITE>(inPriority, true, args) ...};
        constexpr int nbCopies = sizeof(copies)/sizeof(std::vector<SpCurrentCopy>);
        static_assert(sizeof...(Is) == nbCopies-1, "oups");
        for(int idxCopy = 0 ; idxCopy < nbCopies ; ++idxCopy){
            for(const SpCurrentCopy& cp : copies[idxCopy]){
                result[cp.originAdress] = cp;
            }
        }
        return result;
    }

    template <class Tuple, std::size_t... Is>
    std::unordered_map<const void*, SpCurrentCopy> copyIfWriteAndNotDuplicate(const SpPriority& inPriority, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::unordered_map<const void*, SpCurrentCopy> result;
        // Add the handles
        std::vector<SpCurrentCopy> copies[] = {std::vector<SpCurrentCopy>(), coreCopyIfAccess<Tuple, Is, SpDataAccessMode::WRITE>(inPriority, false, args) ...};
        constexpr int nbCopies = sizeof(copies)/sizeof(std::vector<SpCurrentCopy>);
        static_assert(sizeof...(Is) == nbCopies-1, "oups");
        for(int idxCopy = 0 ; idxCopy < nbCopies ; ++idxCopy){
            for(const SpCurrentCopy& cp : copies[idxCopy]){
                result[cp.originAdress] = cp;
            }
        }
        return result;
    }
    
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t IdxData>
    bool coreManageReadDuplicate(Tuple& args){
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

                auto found = copiedHandles.find(ptr);
                if(found != copiedHandles.end()
                        && found->second.usedInRead
                        && accessMode != SpDataAccessMode::READ){
                    assert(std::is_copy_assignable<TargetParamType>::value);

                    SpCurrentCopy& cp = found->second;
                    assert(cp.latestAdress != ptr);
                    assert(cp.latestAdress);
                    assert(cp.hasBeenDeleted == false);
                    this->taskInternal(SpTaskActivation::ENABLE, SpPriority(0),
                                       SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                       [](TargetParamType& output){

                        delete &output;
                    }).setTaskName("delete");

                    copiedHandles.erase(found);
                }

                indexHh += 1;
            }
        }

        return true;
    }

    template <class Tuple, std::size_t... Is>
    void manageReadDuplicate(Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        const bool returnValues[] = {true, coreManageReadDuplicate<Tuple, Is>(args) ...};
        (void)returnValues;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t IdxData>
    std::vector<SpGeneralSpecGroup*> coreGetCorrespondingCopyGroups(Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        std::vector<SpGeneralSpecGroup*> groups;

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

            auto found = copiedHandles.find(ptr);
            if(found != copiedHandles.end()){
                assert(found->second.lastestSpecGroup);
                groups.emplace_back(found->second.lastestSpecGroup);
            }

            indexHh += 1;
        }

        return groups;
    }

    template <class Tuple, std::size_t... Is>
    std::vector<SpGeneralSpecGroup*> getCorrespondingCopyGroups(Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::vector<SpGeneralSpecGroup*> result;
        // Add the handles
        std::vector<SpGeneralSpecGroup*> groups[] = {std::vector<SpGeneralSpecGroup*>(), coreGetCorrespondingCopyGroups<Tuple, Is>(args) ...};
        constexpr int nbRes = sizeof(groups)/sizeof(std::vector<SpGeneralSpecGroup*>);
        static_assert(sizeof...(Is) == nbRes-1, "oups");
        for(int idxRes = 0 ; idxRes < nbRes ; ++idxRes){
            result.insert(result.end(),groups[idxRes].begin(), groups[idxRes].end());
        }

        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t IdxData>
    std::unordered_map<const void*, SpCurrentCopy> coreGetCorrespondingCopyList(Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        std::unordered_map<const void*, SpCurrentCopy> copies;

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
        long int indexHh = 0;
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

            auto found = copiedHandles.find(ptr);
            if(found != copiedHandles.end()){
                assert(found->second.lastestSpecGroup);
                copies[ptr] = (found->second);
            }

            indexHh += 1;
        }

        return copies;
    }

    template <class Tuple, std::size_t... Is>
    std::unordered_map<const void*, SpCurrentCopy> getCorrespondingCopyList(Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        std::unordered_map<const void*, SpCurrentCopy> result;
        // Add the handles
        std::unordered_map<const void*, SpCurrentCopy> groups[] = {std::unordered_map<const void*, SpCurrentCopy>(), coreGetCorrespondingCopyList<Tuple, Is>(args) ...};
        constexpr int nbRes = sizeof(groups)/sizeof(std::unordered_map<const void*, SpCurrentCopy>);
        static_assert(sizeof...(Is) == nbRes-1, "oups");
        for(int idxRes = 0 ; idxRes < nbRes ; ++idxRes){
            result.insert(groups[idxRes].begin(), groups[idxRes].end());
        }

        return result;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////


    template <class Tuple, std::size_t IdxData>
    bool coreRemoveAllCorrespondingCopies(std::unordered_map<const void*,SpCurrentCopy>& copiesList, Tuple& args){
        using ScalarOrContainerType = std::remove_reference_t<typename std::tuple_element<IdxData, Tuple>::type>;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;

        bool removeSomething = false;

        if constexpr (std::is_destructible<TargetParamType>::value){

            auto hh = getDataHandle(scalarOrContainerData);
            assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
            long int indexHh = 0;
            for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

                if(auto found = copiesList.find(ptr); found != copiesList.end()){
                    if(accessMode != SpDataAccessMode::READ){
                        if(found->second.hasBeenDeleted == false){
                            assert(std::is_copy_assignable<TargetParamType>::value);
                            SpCurrentCopy& cp = found->second;
                            assert(cp.latestAdress);
                            this->taskInternal(SpTaskActivation::ENABLE, SpPriority(0),
                                               SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                               [ptr = cp.latestAdress](TargetParamType& output){
                                assert(ptr ==  &output);
                                delete &output;
                            }).setTaskName("delete");
                        }
                        copiesList.erase(found);
                        removeSomething = true;
                     }
// This is not true anymore
//                     else{
//                         assert(found->second.usedInRead == true);
//                     }
                }

                indexHh += 1;
            }
        }

        return removeSomething;
    }

    template <class Tuple, std::size_t... Is>
    void removeAllCorrespondingCopies(Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        bool result[] = {true, coreRemoveAllCorrespondingCopies<Tuple, Is>(copiedHandles, args) ...};
        (void)result;
    }


    template <class Tuple, std::size_t... Is>
    void removeAllCorrespondingCopiesList( std::unordered_map<const void*,SpCurrentCopy>& copiesList, Tuple& args, std::index_sequence<Is...>){
        static_assert(std::tuple_size<Tuple>::value-1 == sizeof...(Is), "Is must be the parameters without the function");
        const bool returnValues[] = {true, coreRemoveAllCorrespondingCopies<Tuple, Is>(copiesList, args) ...};
        (void)returnValues;
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
        const bool results[] = {true, coreAllAreCopiable<Tuple, Is>() ...};
        constexpr int nbHandles = sizeof(results)/sizeof(bool); //parce qu'on ne peut pas directement connaitre la taille du tableua ?
        static_assert(sizeof...(Is) == nbHandles-1, "oups"); // cas ou l'assert serait faux ?
        for(int idxHandle = 0 ; idxHandle < nbHandles ; ++idxHandle){
            if(results[idxHandle] == false){ //si jamais UNE seule des données n'est pas copiable/deletable on renvoie faux
                return false;
            }
        }
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t IdxData, class TaskCorePtr>
    bool coreHandleCreationSpec(Tuple& args, TaskCorePtr& aTask, std::unordered_map<const void*, SpCurrentCopy>& extraCopies){
        using ScalarOrContainerType = typename std::remove_reference<typename std::tuple_element<IdxData, Tuple>::type>::type;
        auto& scalarOrContainerData = std::get<IdxData>(args);

        const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
        using TargetParamType = typename ScalarOrContainerType::RawHandleType;
        static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                      "They should all be default constructible here");

        auto hh = getDataHandle(scalarOrContainerData);
        assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);

        long int indexHh = 0;
        for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
            assert(ptr == getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
            SpDataHandle* h1 = hh[indexHh];

            if(auto found = extraCopies.find(ptr); found != extraCopies.end()){
                const SpCurrentCopy& cp = found->second;
                assert(cp.isDefined());
                assert(cp.originAdress == ptr);
                assert(cp.latestAdress != nullptr);

                SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                const long int handleKey1 = h1copy->addDependence(aTask, accessMode);
                if(indexHh == 0){
                    aTask->template setDataHandle<IdxData>(h1copy, handleKey1);
                }
                else{
                    assert(ScalarOrContainerType::IsScalar == false);
                    aTask->template addDataHandleExtra<IdxData>(h1copy, handleKey1);
                }
                aTask->template updatePtr<IdxData>(indexHh, reinterpret_cast<TargetParamType*>(cp.latestAdress));
            }
            else if(auto found2 = copiedHandles.find(ptr); found2 != copiedHandles.end()){
                const SpCurrentCopy& cp = found2->second;
                assert(cp.isDefined());
                assert(cp.originAdress == ptr);
                assert(cp.latestAdress != nullptr);

                SpDataHandle* h1copy = getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
                const long int handleKey1 = h1copy->addDependence(aTask, accessMode);
                if(indexHh == 0){
                    aTask->template setDataHandle<IdxData>(h1copy, handleKey1);
                }
                else{
                    assert(ScalarOrContainerType::IsScalar == false);
                    aTask->template addDataHandleExtra<IdxData>(h1copy, handleKey1);
                }
                aTask->template updatePtr<IdxData>(indexHh, reinterpret_cast<TargetParamType*>(cp.latestAdress));
            }
            else{
                assert(accessMode == SpDataAccessMode::READ);
                SpDebugPrint() << "accessMode in runtime to add dependence -- => " << SpModeToStr(accessMode);
                const long int handleKey1 = h1->addDependence(aTask, accessMode);
                if(indexHh == 0){
                    aTask->template setDataHandle<IdxData>(h1, handleKey1);
                }
                else{
                    assert(ScalarOrContainerType::IsScalar == false);
                    aTask->template addDataHandleExtra<IdxData>(h1, handleKey1);
                }
            }

            indexHh += 1;
        }
        return true;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////


    template <class... ParamsAndTask>
    SpAbstractTaskWithReturn<void>::SpTaskViewer taskInternal(const SpTaskActivation inActivation, SpPriority inPriority, ParamsAndTask&&... inParamsAndTask){
        auto sequenceParamsNoFunction = std::make_index_sequence<sizeof...(ParamsAndTask)-1>{};
        return coreTaskCreation(inActivation, inPriority, std::forward_as_tuple(inParamsAndTask...), sequenceParamsNoFunction);
    }


    ///////////////////////////////////////////////////////////////////////////
    std::function<bool(int,const SpProbability&)> specFormula;

public:
    void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula){
        specFormula = std::move(inFormula);
    }

    void thisTaskIsReady(SpAbstractTask* aTask) final {
        SpGeneralSpecGroup* specGroup = aTask->getSpecGroup();
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
