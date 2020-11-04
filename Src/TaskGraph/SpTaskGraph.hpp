#ifndef SPTASKGRAPH_HPP
#define SPTASKGRAPH_HPP

#include <functional>
#include <list>
#include <map>
#include <unordered_map>
#include <memory>
#include <unistd.h>
#include <cmath>
#include <type_traits>
#include <initializer_list>

#include "Config/SpConfig.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpAbstractTask.hpp"
#include "Task/SpTask.hpp"
#include "Data/SpDependence.hpp"
#include "Data/SpDataHandle.hpp"
#include "Utils/SpArrayView.hpp"
#include "Utils/SpPriority.hpp"
#include "Utils/SpProbability.hpp"
#include "Utils/SpTimePoint.hpp"
#include "Random/SpMTGenerator.hpp"
#include "Scheduler/SpTaskManager.hpp"
#include "Speculation/SpSpecTaskGroup.hpp"
#include "Utils/small_vector.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "SpAbstractTaskGraph.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Scheduler/SpTaskManagerListener.hpp"

template <const bool isSpeculativeTaskGraph>
class SpTaskGraphCommon : public SpAbstractTaskGraph {
protected:
    //! Map of all data handles
    std::unordered_map<void*, std::unique_ptr<SpDataHandle>> allDataHandles;
    
    ///////////////////////////////////////////////////////////////////////////
    /// Data handle management
    ///////////////////////////////////////////////////////////////////////////

    template <class ParamsType>
    SpDataHandle* getDataHandleCore(ParamsType& inParam){
        auto iterHandleFound = allDataHandles.find(const_cast<std::remove_const_t<ParamsType>*>(&inParam));
        if(iterHandleFound == allDataHandles.end()){
            SpDebugPrint() << "SpTaskGraph -- => Not found";
            // Create a new data handle
            auto newHandle = std::make_unique<SpDataHandle>(const_cast<std::remove_const_t<ParamsType>*>(&inParam));
            // Save the ptr
            SpDataHandle* newHandlePtr = newHandle.get();
            // Move in the map
            allDataHandles[const_cast<std::remove_const_t<ParamsType>*>(&inParam)] = std::move(newHandle);
            // Return the ptr
            return newHandlePtr;
        }
        SpDebugPrint() << "SpTaskGraph -- => Found";
        // Exists, return the pointer
        return iterHandleFound->second.get();
    }

    template <class ParamsType>
    typename std::enable_if_t<ParamsType::IsScalar, small_vector<SpDataHandle*,1>>
    getDataHandle(ParamsType& scalarData){
        static_assert(std::tuple_size<decltype(scalarData.getAllData())>::value, "Size must be one for scalar");
        typename ParamsType::HandleTypePtr ptrToObject = scalarData.getAllData()[0];
        SpDebugPrint() << "SpTaskGraph -- Look for " << ptrToObject << " with mode provided " << SpModeToStr(ParamsType::AccessMode)
                       << " scalarData address " << &scalarData;
        return small_vector<SpDataHandle*, 1>{getDataHandleCore(*ptrToObject)};
    }

    template <class ParamsType>
    typename std::enable_if_t<!ParamsType::IsScalar,small_vector<SpDataHandle*>>
    getDataHandle(ParamsType& containerData){
        small_vector<SpDataHandle*> handles;
        for(typename ParamsType::HandleTypePtr ptrToObject : containerData.getAllData()){
            SpDebugPrint() << "SpTaskGraph -- Look for " << ptrToObject << " with array view in getDataHandleExtra";
            SpDataHandle* handle = getDataHandleCore(*ptrToObject);
            handles.emplace_back(handle);
        }
        return handles;
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Task method call argument partitioning and dispatch 
    ///////////////////////////////////////////////////////////////////////////
      
    template <typename T>
    using is_prio_or_proba = typename std::disjunction<std::is_same<std::decay_t<T>, SpPriority>, std::is_same<std::decay_t<T>, SpProbability>>;

    template <typename T>
    inline static constexpr bool is_prio_or_proba_v = is_prio_or_proba<T>::value;
    
    template <class T>
    using is_data_dependency = std::conjunction<has_getView<T>, has_getAllData<T>>;
    
    template <class T>
    inline static constexpr bool is_data_dependency_v = is_data_dependency<T>::value;
    
    template <class... ParamsTy>
    using contains_maybe_write_dependencies = std::disjunction<access_modes_are_equal<SpDataAccessMode::POTENTIAL_WRITE, ParamsTy>...>;

    template <class... ParamsTy>
    inline static constexpr bool contains_maybe_write_dependencies_v = contains_maybe_write_dependencies<ParamsTy...>::value; 

    template <typename Func, class T0, class T1, class... ParamsTy,
    typename=std::enable_if_t<
             std::conjunction_v<is_prio_or_proba<T0>, is_prio_or_proba<T1>, std::negation<std::is_same<std::decay_t<T0>, std::decay_t<T1>>>>>>    
    auto callWithPartitionedArgs(Func&& f, T0&& t0, T1&& t1, ParamsTy&&... inParams) {
        static_assert(isSpeculativeTaskGraph, "SpTaskGraph::task of non speculative task graph should not be given a probability.");
        if constexpr(std::is_same_v<std::decay_t<T0>, SpProbability>) {
            return callWithPartitionedArgsStage2<true>(std::forward<Func>(f), std::forward<T1>(t1), std::forward<T0>(t0), std::forward<ParamsTy>(inParams)...);
        } else  {
            return callWithPartitionedArgsStage2<true>(std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<ParamsTy>(inParams)...);
        }
    }

    template <typename Func, class T0, class... ParamsTy,
    typename=std::enable_if_t<is_prio_or_proba_v<T0>>>
    auto callWithPartitionedArgs(Func&& f, T0&& t0, ParamsTy&&... inParams) {
        if constexpr(std::is_same_v<std::decay_t<T0>, SpProbability>) {
            static_assert(isSpeculativeTaskGraph, "SpTaskGraph::task of non speculative task graph should not be given a probability.");
            return callWithPartitionedArgsStage2<true>(std::forward<Func>(f), SpPriority(0), std::forward<T0>(t0), std::forward<ParamsTy>(inParams)...);       
        }else {
            return callWithPartitionedArgsStage2<false>(std::forward<Func>(f), std::forward<T0>(t0), SpProbability(), std::forward<ParamsTy>(inParams)...);    
        }    
    }

    template <typename Func, class... ParamsTy>
    auto callWithPartitionedArgs(Func&& f, ParamsTy&&... inParams) {
        static_assert(sizeof...(inParams) > 0, "SpTaskGraph::task should be given at least a callable.");
        return callWithPartitionedArgsStage2<false>(std::forward<Func>(f), SpPriority(0), SpProbability(), std::forward<ParamsTy>(inParams)...);
    }
         
    template <class Tuple, size_t n, typename = std::make_index_sequence<n>>
    struct dispatchRotate {};
    
    template <class Tuple, size_t n, size_t... Is>
    struct dispatchRotate<Tuple, n, std::index_sequence<Is...>> {
        template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class T2, class T3>
        static auto doDispatch(SpTaskGraphCommon& tg, Func&& f, T0&& t0, T1&& t1, typename std::tuple_element<Is, Tuple>::type... args, T2&& t2, T3&& t3) {
            if constexpr(is_data_dependency_v<std::remove_reference_t<T2>>) {
                return tg.callWithPartitionedArgsStage3<probabilityArgWasGivenByUser>(std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                               std::forward<T3>(t3), std::forward<typename std::tuple_element<Is, Tuple>::type>(args)..., std::forward<T2>(t2));
            } else {
                return tg.callWithPartitionedArgsStage3<probabilityArgWasGivenByUser>(std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                                std::forward<T2>(t2), std::forward<T3>(t3), std::forward<typename std::tuple_element<Is, Tuple>::type>(args)...);
            }   
        }
    };
               
    template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class... ParamsAndTask>
    auto callWithPartitionedArgsStage2(Func&& f, T0&& t0, T1&& t1, ParamsAndTask&&... inParamsAndTask){
        static_assert(sizeof...(ParamsAndTask) > 0, "SpTaskGraph::task should be given at least a callable.");
        
        using TupleTy = decltype(std::forward_as_tuple(std::forward<ParamsAndTask>(inParamsAndTask)...));
        
        if constexpr(sizeof...(ParamsAndTask) >= 2) {
            return dispatchRotate<TupleTy, std::tuple_size<TupleTy>::value-2>::template doDispatch<probabilityArgWasGivenByUser>(*this,
                   std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1), std::forward<ParamsAndTask>(inParamsAndTask)...);
        } else {
            return callWithPartitionedArgsStage3<probabilityArgWasGivenByUser>(std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1),
            std::forward<ParamsAndTask>(inParamsAndTask)...);
        }    
    }
    
    template <typename T>
    decltype(auto) wrapIfNotAlreadyWrapped(T&& callable) {
        if constexpr(is_instantiation_of_callable_wrapper_v<std::remove_reference_t<T>>) {
            return std::forward<T>(callable);
        } else {
            return SpCpu(std::forward<T>(callable));
        }
    }
    
    template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class T2, class T3, class... ParamsTy,
    typename = std::enable_if_t<std::conjunction_v<std::negation<is_data_dependency<std::remove_reference_t<T2>>>,
                                                   std::negation<is_data_dependency<std::remove_reference_t<T3>>>>>>
    auto callWithPartitionedArgsStage3(Func&& f, T0&& t0, T1&& t1, T2&& t2, T3&& t3, ParamsTy&&... params) {
        
        auto dispatchStage4 = [&, this](auto&& c1, auto&& c2) {
            return this->callWithPartitionedArgsStage4<probabilityArgWasGivenByUser>(
                    std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                    std::forward<decltype(c1)>(c1), std::forward<decltype(c2)>(c2),
                    std::forward<ParamsTy>(params)...);
        };

        auto&& c1 = wrapIfNotAlreadyWrapped(std::forward<T2>(t2));
        auto&& c2 = wrapIfNotAlreadyWrapped(std::forward<T3>(t3));    

        if constexpr(is_instantiation_of_callable_wrapper_with_type_v<std::remove_reference_t<T2>, SpCallableType::GPU>) {
            dispatchStage4(std::forward<decltype(c2)>(c2), std::forward<decltype(c1)>(c1));
        } else {
            dispatchStage4(std::forward<decltype(c1)>(c1), std::forward<decltype(c2)>(c2));
        }
    }
    
    template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class T2, class... ParamsTy,
    typename = std::enable_if_t<std::negation_v<is_data_dependency<std::remove_reference_t<T2>>>>>
    auto callWithPartitionedArgsStage3(Func&& f, T0&& t0, T1&& t1, T2&& t2, ParamsTy&&... params) {
        auto&& c1 = wrapIfNotAlreadyWrapped(std::forward<T2>(t2));
        
        return callWithPartitionedArgsStage4<probabilityArgWasGivenByUser>(
                std::forward<Func>(f), std::forward<T0>(t0), std::forward<T1>(t1),
                std::forward<decltype(c1)>(c1), std::forward<ParamsTy>(params)...);
    }
    
    template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class T2, class T3, class... ParamsTy,
    typename = std::enable_if_t<std::conjunction_v<is_instantiation_of_callable_wrapper<T2>, is_instantiation_of_callable_wrapper<T3>>>>
    auto callWithPartitionedArgsStage4(Func&& f, T0&& t0, T1&& t1, T2&& t2, [[maybe_unused]] T3&& t3, ParamsTy&&...params) {
        static_assert(std::conjunction_v<is_instantiation_of_callable_wrapper_with_type<std::remove_reference_t<T2>, SpCallableType::CPU>,
                                       is_instantiation_of_callable_wrapper_with_type<std::remove_reference_t<T3>, SpCallableType::GPU>>,
                      "SpTaskGraph::task when providing two callables to a task one should be a CPU callable and the other a GPU callable");
        
        static_assert(std::conjunction_v<has_getView<ParamsTy>..., has_getAllData<ParamsTy>...>,
                      "SpTaskGraph::task some data dependencies don't have a getView() and/or a getAllData method.");
        
        static_assert(std::is_invocable_v<decltype(t2.getCallableRef()), decltype(params.getView())...>,
                      "SpTaskGraph::task Cpu callable is not invocable with data dependencies.");

        constexpr bool isPotentialTask = contains_maybe_write_dependencies_v<ParamsTy...>; 
        
        static_assert(isSpeculativeTaskGraph || !isPotentialTask, "SpTaskGraph::task of non speculative task graph should not be given maybe-write data dependencies.");
        
        static_assert(!(probabilityArgWasGivenByUser && !isPotentialTask),
                      "SpTaskGraph::task no probability should be specified for normal tasks.");
                      
        auto dataDepTuple = std::forward_as_tuple(std::forward<ParamsTy>(params)...); 
        auto callableTuple = 
        [&](){
            if constexpr(SpConfig::CompileWithCuda) {
                static_assert(std::is_invocable_v<decltype(t3.getCallableRef()), decltype(params.getView())...>,
                      "SpTaskGraph::task Gpu callable is not invocable with data dependencies.");
                return std::forward_as_tuple(std::forward<T2>(t2), std::forward<T3>(t3));
            } else {
                return std::forward_as_tuple(std::forward<T2>(t2));
            }
        }();
        
        if constexpr(isSpeculativeTaskGraph) {
            return std::invoke(std::forward<Func>(f), std::bool_constant<isPotentialTask>{}, std::forward<T0>(t0), std::forward<T1>(t1), dataDepTuple, callableTuple);
        } else {
            return std::invoke(std::forward<Func>(f), std::forward<T0>(t0), dataDepTuple, callableTuple);
        }
    }

    template <bool probabilityArgWasGivenByUser, typename Func, class T0, class T1, class T2, class... ParamsTy>
    auto callWithPartitionedArgsStage4(Func&& f, T0&& t0, T1&& t1, T2&& t2, ParamsTy&&...params) {
        static_assert(!(!SpConfig::CompileWithCuda && is_instantiation_of_callable_wrapper_with_type_v<std::remove_reference_t<T2>, SpCallableType::GPU>),
                      "SpTaskGraph::task : SPETABARU_USE_CUDA macro is undefined. Unable to compile tasks for which only a GPU callable has been provided.");

        static_assert(std::conjunction_v<has_getView<ParamsTy>..., has_getAllData<ParamsTy>...>,
                      "SpTaskGraph::task some data dependencies don't have a getView() and/or a getAllData method.");
        
        static_assert(std::is_invocable_v<decltype(t2.getCallableRef()), decltype(params.getView())...>,
                      "SpTaskGraph::task callable is not invocable with data dependencies.");
        
        constexpr bool isPotentialTask = contains_maybe_write_dependencies_v<ParamsTy...>; 
        
        static_assert(isSpeculativeTaskGraph || !isPotentialTask, "SpTaskGraph::task of non speculative task graph should not be given maybe-write data dependencies.");
        
        static_assert(!(probabilityArgWasGivenByUser && !isPotentialTask),
                      "SpTaskGraph::task no probability should be specified for normal tasks.");

        auto dataDepTuple = std::forward_as_tuple(std::forward<ParamsTy>(params)...); 
        auto callableTuple = std::forward_as_tuple(std::forward<T2>(t2));
        
        if constexpr(isSpeculativeTaskGraph) {
            return std::invoke(std::forward<Func>(f), std::bool_constant<isPotentialTask>{}, std::forward<T0>(t0), std::forward<T1>(t1), dataDepTuple, callableTuple);
        } else {
            return std::invoke(std::forward<Func>(f), std::forward<T0>(t0), dataDepTuple, callableTuple);
        }
    }
    
    template <typename CallableTy, typename TupleTy>
    static auto apply_on_data_dep_tuple(CallableTy&& c, TupleTy&& t) {
        return std::apply([&c](auto&&... elt) {
                            return std::invoke(std::forward<CallableTy>(c), std::forward<decltype(elt)>(elt).getView()...); 
                          }, std::forward<TupleTy>(t));
    }
    
    explicit SpTaskGraphCommon() : allDataHandles() {}
    
    // No copy and no move
    SpTaskGraphCommon(const SpTaskGraphCommon&) = delete;
    SpTaskGraphCommon(SpTaskGraphCommon&&) = delete;
    SpTaskGraphCommon& operator=(const SpTaskGraphCommon&) = delete;
    SpTaskGraphCommon& operator=(SpTaskGraphCommon&&) = delete;
};

template <SpSpeculativeModel SpecModel = SpSpeculativeModel::SP_MODEL_1>
class SpTaskGraph : public SpTaskGraphCommon<true>, public SpTaskManagerListener {
    
    static_assert(SpecModel == SpSpeculativeModel::SP_MODEL_1
                   || SpecModel == SpSpeculativeModel::SP_MODEL_2
                   || SpecModel == SpSpeculativeModel::SP_MODEL_3, "Should not happen");
    
//=====----------------------------=====
//         Private API - members
//=====----------------------------=====
private:

    class SpAbstractDeleter{
    public:
        virtual ~SpAbstractDeleter(){}
        virtual void deleteObject(void* ptr) = 0;
        virtual void createDeleteTaskForObject(SpTaskGraph& tg, void* ptr) = 0;
    };

    template <class ObjectType>
    class SpDeleter : public SpAbstractDeleter{
    public:
        ~SpDeleter() = default;

        void deleteObject(void* ptr) override final{
            delete reinterpret_cast<ObjectType*>(ptr);
        }
        
        void createDeleteTaskForObject(SpTaskGraph<SpecModel>& tg, void* ptr) override final{
            tg.taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(tg.emptyCopyMap)}, SpTaskActivation::ENABLE, SpPriority(0),
                            SpWrite(*reinterpret_cast<ObjectType*>(ptr)),
                            [](ObjectType& output){
                                delete &output;
                            }).setTaskName("sp-delete");
        }
    };

    struct SpCurrentCopy{
        SpCurrentCopy()
            : originAdress(nullptr), sourceAdress(nullptr), latestAdress(nullptr), lastestSpecGroup(nullptr),
              latestCopyTask(nullptr), usedInRead(false), isUniquePtr(std::make_shared<bool>(true)) {
        }

        SpCurrentCopy(void* inOriginAddress, void* inSourceAddress, void* inLatestAddress, SpGeneralSpecGroup<SpecModel>* inLatestSpecGroup,
                      SpAbstractTask* inLatestCopyTask, std::shared_ptr<SpAbstractDeleter> inDeleter)
                      : originAdress(inOriginAddress), sourceAdress(inSourceAddress), latestAdress(inLatestAddress), lastestSpecGroup(inLatestSpecGroup),
                        latestCopyTask(inLatestCopyTask), usedInRead(false), isUniquePtr(std::make_shared<bool>(true)), deleter(inDeleter) {} 

        bool isDefined() const{
            return originAdress != nullptr;
        }
        
        void* originAdress;
        void* sourceAdress;
        void* latestAdress;
        SpGeneralSpecGroup<SpecModel>* lastestSpecGroup;
        SpAbstractTask* latestCopyTask;
        bool usedInRead;
        std::shared_ptr<bool> isUniquePtr;
        std::shared_ptr<SpAbstractDeleter> deleter;
    };
    
    using CopyMapTy = std::unordered_map<const void*, SpCurrentCopy>;
    using CopyMapPtrTy = CopyMapTy*;
       
    //! Current speculation group
    SpGeneralSpecGroup<SpecModel>* currentSpecGroup;

    //! List of all speculation groups that have been created 
    std::list<std::unique_ptr<SpGeneralSpecGroup<SpecModel>>> specGroups;
    
    //! Mutex for spec group list
    std::mutex specGroupMutex;
    
    using ExecutionPathWeakPtrTy = std::weak_ptr<small_vector<CopyMapTy>>;
    using ExecutionPathSharedPtrTy = std::shared_ptr<small_vector<CopyMapTy>>;
    
    //! Map mapping original addresses to execution paths
    std::unordered_map<const void*, ExecutionPathSharedPtrTy> hashMap;
    
    //! Predicate function used to decide if a speculative task and any of its 
    //  dependent speculative tasks should be allowed to run
    std::function<bool(int,const SpProbability&)> specFormula;

    //! Empty map of copies
    CopyMapTy emptyCopyMap;

//=====----------------------------=====
//         Private API - methods
//=====----------------------------=====
private:

    ///////////////////////////////////////////////////////////////////////////
    /// Cleanup 
    ///////////////////////////////////////////////////////////////////////////
    
    void releaseCopies(small_vector_base<CopyMapTy>& copyMaps){
        for(auto& copyMapIt : copyMaps){
            for(auto& iter : copyMapIt) {
                assert(iter.second.latestAdress);
                assert(iter.second.deleter);
                if(iter.second.isUniquePtr.use_count() == 1) {
                    iter.second.deleter->deleteObject(iter.second.latestAdress);
                }
            }
            copyMapIt.clear();
        }
        copyMaps.clear();
    }
    
    template <bool isSpeculative, class TaskCoreTy, std::size_t N>
    void setDataHandlesOfTaskGeneric(TaskCoreTy* aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
        auto& args = aTask->getDataDependencyTupleRef();
        SpUtils::foreach_in_tuple(
            [&, this, aTask](auto index, auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                static_assert(!isSpeculative || (std::is_default_constructible_v<TargetParamType> && std::is_copy_assignable_v<TargetParamType>),
                              "They should all be default constructible here");
                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;

                auto hh = this->getDataHandle(scalarOrContainerData);
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                long int indexHh = 0;
                
                for([[maybe_unused]] typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    SpDataHandle* h = hh[indexHh];
                    
                    bool foundAddressInCopies = false;
                    [[maybe_unused]] void *cpLatestAddress = nullptr;
                    
                    if constexpr(isSpeculative){
                        for(auto me : copyMapsToLookInto) {
                            if(auto found = me->find(ptr); found != me->end()){
                                const SpCurrentCopy& cp = found->second;
                                assert(cp.isDefined());
                                assert(cp.originAdress == ptr);
                                assert(cp.latestAdress != nullptr);
                                h = this->getDataHandleCore(*reinterpret_cast<TargetParamType*>(cp.latestAdress));
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
                        aTask->template setDataHandle<index>(h, handleKey);
                    }
                    else{
                        assert(ScalarOrContainerType::IsScalar == false);
                        aTask->template addDataHandleExtra<index>(h, handleKey);
                    }
                    
                    if constexpr(isSpeculative) {
                        if(foundAddressInCopies) {
                            aTask->template updatePtr<index>(indexHh, reinterpret_cast<TargetParamType*>(cpLatestAddress));
                        }
                    }
                    
                    indexHh += 1;
                }
            }
        , args);
    }
    
    template <class TaskCoreTy, std::size_t N>
    inline void setDataHandlesOfTask(TaskCoreTy* aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
        setDataHandlesOfTaskGeneric<false>(aTask, copyMapsToLookInto);
    }

    template <class TaskCoreTy, std::size_t N>
    inline void setDataHandlesOfTaskAndUpdateDataDepTupleOfTask(TaskCoreTy* aTask, const std::array<CopyMapPtrTy, N>& copyMapsToLookInto){
        setDataHandlesOfTaskGeneric<true>(aTask, copyMapsToLookInto);
    }
  
    ///////////////////////////////////////////////////////////////////////////
    /// Core task creation and task submission to scheduler
    ///////////////////////////////////////////////////////////////////////////
    
    //! Convert tuple to data and call the function
    //! Args is a value to allow for move or pass a rvalue reference
    template <template<typename...> typename TaskType, const bool isSpeculative, class DataDependencyTupleTy, class CallableTupleTy,
             std::size_t N, typename... T>
    auto coreTaskCreationAux(const SpTaskActivation inActivation, const SpPriority& inPriority, DataDependencyTupleTy& dataDepTuple,
                             CallableTupleTy& callableTuple, [[maybe_unused]] const std::array<CopyMapPtrTy, N>& copyMapsToLookInto, T... additionalArgs){
        SpDebugPrint() << "SpTaskGraph -- coreTaskCreation";
        
        auto createCopyTupleFunc =
        [](auto& tupleOfRefs) {
            return [&tupleOfRefs]() {
                // Notice we create a tuple from lvalue references (we don't perfect forward to the
                // std::tuple_element_t<Is, std::remove_reference_t<DataDependencyTupleTy> type) because we want
                // to copy all data dependency objects into the new tuple (i.e. we don't want to move from the object even 
                // if it is an rvalue because we might need to reference it again later on when we insert yet another
                // speculative version of the same task).
                return std::apply([](const auto&...elt) {
                    return std::make_tuple(elt...);
                }, tupleOfRefs);
            };
        };
               
        auto dataDependencyTupleCopyFunc = createCopyTupleFunc(dataDepTuple);
        auto callableTupleCopyFunc = createCopyTupleFunc(callableTuple);
                                            
        static_assert(0 < std::tuple_size<decltype(callableTupleCopyFunc())>::value );
         
        using DataDependencyTupleCopyTy = std::remove_reference_t<decltype(dataDependencyTupleCopyFunc())>;    
        using CallableTupleCopyTy = std::remove_reference_t<decltype(callableTupleCopyFunc())>;
        using RetTypeRef = decltype(apply_on_data_dep_tuple(std::get<0>(callableTuple).getCallableRef(), dataDepTuple));
        using RetType = std::remove_reference_t<RetTypeRef>;
        using TaskTy = TaskType<RetType, DataDependencyTupleCopyTy, CallableTupleCopyTy>;

        // Create a task with a copy of the callables and the data dependency objects
        auto aTask = new TaskTy(this, inActivation, inPriority, dataDependencyTupleCopyFunc(), callableTupleCopyFunc(), additionalArgs...);

        // Lock the task
        aTask->takeControl();
        
        // Add the handles
        if constexpr(!isSpeculative) {
            setDataHandlesOfTask(aTask, copyMapsToLookInto);
        } else {
            setDataHandlesOfTaskAndUpdateDataDepTupleOfTask(aTask, copyMapsToLookInto);
        }

        // Check coherency
        assert(aTask->areDepsCorrect());

        // The task has been initialized
        aTask->setState(SpTaskState::INITIALIZED);

        // Get the view
        auto descriptor = aTask->getViewer();

        aTask->setState(SpTaskState::WAITING_TO_BE_READY);
        
        if(currentSpecGroup){
            currentSpecGroup->addCopyTask(aTask);
            aTask->setSpecGroup(currentSpecGroup);
        }
        
        aTask->releaseControl();

        SpDebugPrint() << "SpTaskGraph -- coreTaskCreation => " << aTask << " of id " << aTask->getId();
        
        // Push to the scheduler
        this->scheduler.addNewTask(aTask);
        
        // Return the view
        return descriptor;
    }
    
    template <std::size_t N, class DataDependencyTupleTy, class CallableTupleTy>
    inline auto coreTaskCreation(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto, 
                                 const SpTaskActivation inActivation, 
                                 const SpPriority& inPriority, 
                                 DataDependencyTupleTy&& dataDepTuple, 
                                 CallableTupleTy&& callableTuple){
        return coreTaskCreationAux<SpTask, false>(inActivation, inPriority,
               std::forward<DataDependencyTupleTy>(dataDepTuple), std::forward<CallableTupleTy>(callableTuple), copyMapsToLookInto);
    }
    
    template <std::size_t N, class DataDependencyTupleTy, class CallableTupleTy>
    inline auto coreTaskCreationSpeculative(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                            const SpTaskActivation inActivation,
                                            const SpPriority& inPriority,
                                            DataDependencyTupleTy&& dataDepTuple,
                                            CallableTupleTy&& callableTuple) {
        return coreTaskCreationAux<SpTask, true>(inActivation, inPriority,
               std::forward<DataDependencyTupleTy>(dataDepTuple), std::forward<CallableTupleTy>(callableTuple), copyMapsToLookInto);
    }
         
    ///////////////////////////////////////////////////////////////////////////
    /// Core logic (task insertion for the different speculative models)
    ///////////////////////////////////////////////////////////////////////////
            
    template <typename T>
    struct SpRange{
        T beginIt;
        T endIt;
        T currentIt;
        SpRange(T inBeginIt, T inEndIt, T inCurrentIt) : beginIt(inBeginIt), endIt(inEndIt), currentIt(inCurrentIt) {}
    };
    
    template <const bool isPotentialTask, class CallableTupleTy, class DataDependencyTupleTy>
    auto preCoreTaskCreation(const SpPriority& inPriority, const SpProbability& inProbability, DataDependencyTupleTy&& dataDepTuple, CallableTupleTy&& callableTuple) {
        std::unique_lock<std::mutex> lock(specGroupMutex);
        
        bool isPathResultingFromMerge = false;
        auto executionPaths = getCorrespondingExecutionPaths(std::forward<DataDependencyTupleTy>(dataDepTuple));
    
        ExecutionPathSharedPtrTy e;
        
        small_vector<const void *> originalAddresses;
        
        constexpr const unsigned char maybeWriteFlags = 1 << static_cast<unsigned char>(SpDataAccessMode::POTENTIAL_WRITE);
        
        constexpr const unsigned char writeFlags = 1 << static_cast<unsigned char>(SpDataAccessMode::WRITE)
                                                 | 1 << static_cast<unsigned char>(SpDataAccessMode::COMMUTATIVE_WRITE)
                                                 | 1 << static_cast<unsigned char>(SpDataAccessMode::PARALLEL_WRITE);
        
        auto originalAddressesOfMaybeWrittenHandles = getOriginalAddressesOfHandlesWithAccessModes<maybeWriteFlags>(std::forward<DataDependencyTupleTy>(dataDepTuple));
        auto originalAddressesOfWrittenHandles = getOriginalAddressesOfHandlesWithAccessModes<writeFlags>(std::forward<DataDependencyTupleTy>(dataDepTuple));
        
        if(executionPaths.empty()) {
            e = std::make_shared<small_vector<CopyMapTy>>();
        }else if(executionPaths.size() == 1){
            e = executionPaths[0].lock();
        }else{ // merge case
            isPathResultingFromMerge = true;
            e = std::make_shared<small_vector<CopyMapTy>>();
            
            using ExecutionPathIteratorVectorTy = small_vector<SpRange<typename small_vector_base<CopyMapTy>::iterator>>;
            
            ExecutionPathIteratorVectorTy vectorExecutionPaths;
            
            for(auto &ep : executionPaths) {
                vectorExecutionPaths.push_back({ep.lock()->begin(), ep.lock()->end(), ep.lock()->begin()});
            }
            
            auto it = vectorExecutionPaths.begin();
            
            while(true) {
                
                e->emplace_back();
                auto speculationBranchIt = e->end()-1;
            
                for(auto& ep : vectorExecutionPaths) {
                    if(ep.currentIt != ep.endIt) {
                        for(auto mIt = ep.currentIt->cbegin(); mIt != ep.currentIt->cend();) {
                            if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                                if(isUsedByTask(mIt->first, std::forward<DataDependencyTupleTy>(dataDepTuple))) {
                                    (*speculationBranchIt)[mIt->first] = mIt->second;
                                    originalAddresses.push_back(mIt->first);
                                    mIt++;
                                }else{
                                    originalAddresses.push_back(mIt->first);
                                    mIt->second.deleter->createDeleteTaskForObject(*this, mIt->second.latestAdress);
                                    mIt = ep.currentIt->erase(mIt);
                                }
                            }else {
                                (*speculationBranchIt)[mIt->first] = mIt->second;
                                originalAddresses.push_back(mIt->first);
                                mIt++;
                            }
                        }
                    }
                }
                
                if(speculationBranchIt->empty()) {
                    e->pop_back();
                }
                
                if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                    break;
                }
                
                while(it != vectorExecutionPaths.end() && it->currentIt == it->endIt) {
                    it->currentIt = it->beginIt;
                    it++;
                }
                
                if(it != vectorExecutionPaths.end()) {
                    it->currentIt++;
                    it = vectorExecutionPaths.begin();
                }else {
                    break;
                }
            }
            
            std::sort(originalAddresses.begin(), originalAddresses.end());
            originalAddresses.erase(std::unique(originalAddresses.begin(), originalAddresses.end()), originalAddresses.end());
            
            if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_1) {
                setExecutionPathForOriginalAddressesInHashMap(e, originalAddresses);
            }else if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                removeOriginalAddressesFromHashMap(originalAddresses);
            }
        }
        
        if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_2) {
            removeOriginalAddressesFromHashMap(originalAddresses);
        }
        
        auto res = preCoreTaskCreationAux<isPotentialTask>(isPathResultingFromMerge, *e, inPriority, inProbability,
                                                           std::forward<DataDependencyTupleTy>(dataDepTuple), std::forward<CallableTupleTy>(callableTuple));
        
        if constexpr(isPotentialTask) {
            setExecutionPathForOriginalAddressesInHashMap(e, originalAddressesOfMaybeWrittenHandles);
        }
        
        if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_1) {
            removeOriginalAddressesFromHashMap(originalAddressesOfWrittenHandles);
        }else if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
            if(executionPaths.size() == 1) {
                removeOriginalAddressesFromHashMap(originalAddressesOfWrittenHandles);
            }
        }
    
        return res;
    }
    
    template <const bool isPotentialTask, class DataDependencyTupleTy, class CallableTupleTy>
    auto preCoreTaskCreationAux([[maybe_unused]] bool pathResultsFromMerge, small_vector_base<CopyMapTy> &copyMaps, const SpPriority& inPriority,
                                const SpProbability& inProbability, DataDependencyTupleTy &&dataDepTuple, CallableTupleTy &&callableTuple) {
        
        static_assert(SpecModel == SpSpeculativeModel::SP_MODEL_1
                      || SpecModel == SpSpeculativeModel::SP_MODEL_2
                      || SpecModel == SpSpeculativeModel::SP_MODEL_3, "Should not happen");
        
        auto it = copyMaps.begin();
        
        if constexpr (!isPotentialTask && allAreCopieableAndDeletable<DataDependencyTupleTy>() == false){
            for(; it != copyMaps.end(); it++){
                if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_2) {
                    for(auto& e : *it) {
                        e.second.deleter->createDeleteTaskForObject(*this, e.second.latestAdress);
                    }
                    it->clear();
                }else {
                    manageReadDuplicate(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    removeAllCorrespondingCopies(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(pathResultsFromMerge) {
                            removeAllCopiesReadFrom(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                        }
                    }
                }
            }
            return coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                    SpTaskActivation::ENABLE,
                                    inPriority,
                                    std::forward<DataDependencyTupleTy>(dataDepTuple),
                                    std::forward<CallableTupleTy>(callableTuple));
        } else {
            static_assert(allAreCopieableAndDeletable<DataDependencyTupleTy>() == true,
                          "Add data passed to a potential task must be copiable");
            
            using TaskViewTy = decltype(coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                                         std::declval<SpTaskActivation>(), inPriority,
                                                         std::forward<DataDependencyTupleTy>(dataDepTuple),
                                                         std::forward<CallableTupleTy>(callableTuple)));
            
            small_vector<TaskViewTy> speculativeTasks;
            small_vector<std::function<void()>> selectTaskCreationFunctions;
            
            std::shared_ptr<std::atomic<size_t>> numberOfSpeculativeSiblingSpecGroupsCounter = std::make_shared<std::atomic<size_t>>(0);
            
            for(; it != copyMaps.end(); ++it) {
                
                std::unordered_map<const void*, SpCurrentCopy> l1, l2, l1p;
                
                auto groups = getCorrespondingCopyGroups(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                bool oneGroupDisableOrFailed = false;
                
                for(auto gp : groups){
                    if(gp->isSpeculationDisable() || gp->didSpeculationFailed() || gp->didParentSpeculationFailed()){
                        oneGroupDisableOrFailed = true;
                        break;
                    }
                }
                
                const bool taskAlsoSpeculateOnOther = (groups.size() != 0 && !oneGroupDisableOrFailed);
                
                if(!taskAlsoSpeculateOnOther) {
                    manageReadDuplicate(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    removeAllCorrespondingCopies(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(pathResultsFromMerge) {
                            removeAllCopiesReadFrom(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                        }
                    }
                } else {
                    (*numberOfSpeculativeSiblingSpecGroupsCounter)++;
                    std::unique_ptr<SpGeneralSpecGroup<SpecModel>> specGroupSpecTask = std::make_unique<SpGeneralSpecGroup<SpecModel>>
                                                                                       (
                                                                                        true,
                                                                                        numberOfSpeculativeSiblingSpecGroupsCounter
                                                                                       );
                    specGroupSpecTask->addParents(groups);
                    
                    if constexpr(isPotentialTask) {
                        specGroupSpecTask->setProbability(inProbability);
                        currentSpecGroup = specGroupSpecTask.get();
                        l1 = copyIfMaybeWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)},
                                                                         specGroupSpecTask->getActivationStateForCopyTasks(),
                                                                         inPriority,
                                                                         std::forward<DataDependencyTupleTy>(dataDepTuple));
                        assert(taskAlsoSpeculateOnOther == true || l1.size());
                        currentSpecGroup = nullptr;
                    }
                    
                    currentSpecGroup = specGroupSpecTask.get();
                    l2 = copyIfWriteAndNotDuplicateOrUsedInRead(std::array<CopyMapPtrTy, 1>{std::addressof(*it)},
                                                                specGroupSpecTask->getActivationStateForCopyTasks(),
                                                                inPriority,
                                                                std::forward<DataDependencyTupleTy>(dataDepTuple));
                    currentSpecGroup = nullptr;
                    
                    if constexpr(isPotentialTask && SpecModel != SpSpeculativeModel::SP_MODEL_2) {
                        currentSpecGroup = specGroupSpecTask.get();
                        l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)},
                                                           specGroupSpecTask->getActivationStateForCopyTasks(),
                                                           inPriority, std::forward<DataDependencyTupleTy>(dataDepTuple));
                        currentSpecGroup = nullptr;
                    }
                    
                    TaskViewTy taskViewSpec = coreTaskCreationSpeculative(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)},
                                                                          specGroupSpecTask->getActivationStateForSpeculativeTask(),
                                                                          inPriority,
                                                                          std::forward<DataDependencyTupleTy>(dataDepTuple),
                                                                          std::forward<CallableTupleTy>(callableTuple));

                    specGroupSpecTask->setSpecTask(taskViewSpec.getTaskPtr());
                    taskViewSpec.getTaskPtr()->setSpecGroup(specGroupSpecTask.get());
                    speculativeTasks.push_back(taskViewSpec);
                    
                    auto functions = createSelectTaskCreationFunctions(std::array<CopyMapPtrTy, 3>{std::addressof(l1), std::addressof(l2), std::addressof(*it)},
                                                                       specGroupSpecTask.get(),
                                                                       inPriority,
                                                                       std::forward<DataDependencyTupleTy>(dataDepTuple));
                    
                    selectTaskCreationFunctions.reserve(selectTaskCreationFunctions.size() + functions.size());
                    selectTaskCreationFunctions.insert(selectTaskCreationFunctions.end(), functions.begin(), functions.end());
                    
                    manageReadDuplicate(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    removeAllCorrespondingCopies(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                    
                    if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                        if(pathResultsFromMerge) {
                            removeAllCopiesReadFrom(*it, std::forward<DataDependencyTupleTy>(dataDepTuple));
                        }
                    }
                    
                    specGroups.emplace_back(std::move(specGroupSpecTask));
                    
                    if constexpr(isPotentialTask && SpecModel != SpSpeculativeModel::SP_MODEL_2) {
                        it->merge(l1p);
                    }
                }
            }
                
            if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                if(copyMaps.empty()) {
                    copyMaps.emplace_back();
                    it = copyMaps.end()-1;
                } else {
                    it = copyMaps.begin();
                }
            }else {
                copyMaps.emplace_back();
                it = copyMaps.end()-1;
            }
            
            if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_2) {
                for(auto& e : *it) {
                    e.second.deleter->createDeleteTaskForObject(*this, e.second.latestAdress);
                }
                it->clear();
            }
            
            std::unique_ptr<SpGeneralSpecGroup<SpecModel>> specGroupNormalTask = std::make_unique<SpGeneralSpecGroup<SpecModel>>
                                                                                 (
                                                                                    !speculativeTasks.empty(),
                                                                                    numberOfSpeculativeSiblingSpecGroupsCounter
                                                                                 );
            
            specGroupNormalTask->setSpeculationActivation(true);
            
            if constexpr(isPotentialTask) {
                specGroupNormalTask->setProbability(inProbability);
                
                std::unordered_map<const void*, SpCurrentCopy> l1p;
                
                currentSpecGroup = specGroupNormalTask.get();
                if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_1) {
                    if(speculativeTasks.empty()) {
                        l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                                           specGroupNormalTask->getActivationStateForCopyTasks(),
                                                           inPriority,
                                                           std::forward<DataDependencyTupleTy>(dataDepTuple));
                        it->merge(l1p);
                    }
                }else{
                    l1p = copyIfMaybeWriteAndDuplicate(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                                       specGroupNormalTask->getActivationStateForCopyTasks(),
                                                       inPriority,
                                                       std::forward<DataDependencyTupleTy>(dataDepTuple));
                    it->merge(l1p);
                }
                currentSpecGroup = nullptr;
            }
            
            TaskViewTy result = coreTaskCreation(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                                 specGroupNormalTask->getActivationStateForMainTask(),
                                                 inPriority,
                                                 std::forward<DataDependencyTupleTy>(dataDepTuple),
                                                 std::forward<CallableTupleTy>(callableTuple));
            
            specGroupNormalTask->setMainTask(result.getTaskPtr());
            result.getTaskPtr()->setSpecGroup(specGroupNormalTask.get());
            
            for(auto& t : speculativeTasks) {
                SpGeneralSpecGroup<SpecModel>* sg = t.getTaskPtr()->template getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
                sg->setMainTask(result.getTaskPtr());
                t.setOriginalTask(result.getTaskPtr());
            }
            
            for(auto& f : selectTaskCreationFunctions) {
                f();
            }
            
            if constexpr(isPotentialTask) {
                for(auto& t : speculativeTasks) {
                    SpGeneralSpecGroup<SpecModel>* sg = t.getTaskPtr()->template getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
                    addCallbackToTask(t, sg);
                }
                
                addCallbackToTask(result, specGroupNormalTask.get());
            }
            
            specGroups.emplace_back(std::move(specGroupNormalTask));
            
            return result;
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Select task creation
    ///////////////////////////////////////////////////////////////////////////

    template <class Tuple, std::size_t N>
    auto createSelectTaskCreationFunctions(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                               [[maybe_unused]] SpGeneralSpecGroup<SpecModel>* sg,
                                               const SpPriority& inPriority,
                                               Tuple& args){
        small_vector<std::function<void()>> res;
        
        SpUtils::foreach_in_tuple(
            [&, this](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;

                using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                static_assert(std::is_default_constructible<TargetParamType>::value && std::is_copy_assignable<TargetParamType>::value,
                              "They should all be default constructible here");
                              
                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;

                auto hh = this->getDataHandle(scalarOrContainerData);
                
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                
                long int indexHh = 0;
                
                for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    [[maybe_unused]] SpDataHandle* h1 = hh[indexHh];
                    for(auto me : copyMapsToLookInto) {
                        if(auto found = me->find(ptr); found != me->end()){
                            const SpCurrentCopy& cp = found->second;
                            assert(cp.isDefined());
                            assert(cp.originAdress == ptr);
                            assert(cp.latestAdress != nullptr);

                            assert(accessMode == SpDataAccessMode::READ || found->second.usedInRead == false);
                            
                            if constexpr(accessMode != SpDataAccessMode::READ){
                                const bool isCarryingSurelyWrittenValuesOver = accessMode == SpDataAccessMode::WRITE;
                                void* const cpLatestAddress = cp.latestAdress;
                                
                                auto s = [this, &inPriority, h1, cpLatestAddress, sg]() {
                                
                                    SpDataHandle* h1copy = this->getDataHandleCore(*reinterpret_cast<TargetParamType*>(cpLatestAddress));
                                    
                                    auto taskViewSelect = this->taskInternalSpSelect(
                                                       std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
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
                                    auto taskViewDelete = this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(emptyCopyMap)},
                                                                       SpTaskActivation::ENABLE,
                                                                       SpPriority(0),
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
                            } else{
                               found->second.usedInRead = true;
                            }
                            
                            break;
                        }
                        
                    }
                    
                    indexHh += 1;
                }
        }
        , args);
        
        return res;
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Copy task creation
    ///////////////////////////////////////////////////////////////////////////
   
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

        SpDebugPrint() << "SpTaskGraph -- coreCopyCreationCore -- setup copy from " << sourcePtr << " to " << &ptr;
        SpAbstractTaskWithReturn<void>::SpTaskViewer taskView = taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMapToLookInto)},
                                                                            initialActivationState,
                                                                            inPriority,
                                                                            SpWrite(*ptr),
                                                                            SpRead(*sourcePtr),
                                                                            [](TargetParamType& output, const TargetParamType& input){
                SpDebugPrint() << "SpTaskGraph -- coreCopyCreationCore -- execute copy from " << &input << " to " << &output;
                output = input;
        });
        
        taskView.setTaskName("sp-copy");

        return SpCurrentCopy(
                             const_cast<TargetParamType*>(originPtr),       // .originAdress
                             const_cast<TargetParamType*>(sourcePtr),       // .sourceAdress
                             ptr,                                           // .latestAdress
                             currentSpecGroup,                              // .lastestSpecGroup
                             taskView.getTaskPtr(),                         // .latestCopyTask
                             std::make_shared<SpDeleter<TargetParamType>>() // .deleter
                            );
    }
        
    template <bool copyIfAlreadyDuplicate, bool copyIfUsedInRead, SpDataAccessMode targetMode, class Tuple, std::size_t... Is, std::size_t N>
    auto copyIfAccess(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                      const SpTaskActivation initialActivationState,
                      const SpPriority& inPriority,
                      const Tuple& args){
        static_assert(N > 0, "coreCopyIfAccess -- You must provide at least one copy map for inspection.");

        std::unordered_map<const void*, SpCurrentCopy> res;

        SpUtils::foreach_in_tuple(
            [&, this, initialActivationState](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
                
                if constexpr (accessMode == targetMode){
                    static_assert(std::is_default_constructible<TargetParamType>::value
                                  && std::is_copy_assignable<TargetParamType>::value,
                                  "Data must be copiable");

                    auto hh = this->getDataHandle(scalarOrContainerData);
                    assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);

                    long int indexHh = 0;
                    for([[maybe_unused]] typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                        assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                        assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                        SpDataHandle* h1 = hh[indexHh];
                        
                        bool doCopy = false;
                        std::unordered_map<const void*, SpCurrentCopy>* mPtr = nullptr;
                        
                        if constexpr (copyIfAlreadyDuplicate) { // always copy regardless of the fact that the data might have been previously copied
                            doCopy = true;
                            for(auto me : copyMapsToLookInto) {
                                mPtr = me;
                                if(me->find(h1->castPtr<TargetParamType>()) != me->end()) {
                                    break;
                                }
                            }
                        }else if constexpr (copyIfUsedInRead){ // if data has already been previously copied then only copy if the previous copy is used in read
                            bool hasBeenFound = false;
                            for(auto me : copyMapsToLookInto) {
                                mPtr = me;
                                
                                if(auto found = me->find(h1->castPtr<TargetParamType>()); found != me->end()) {
                                    hasBeenFound = true;
                                    doCopy = found->second.usedInRead || found->second.isUniquePtr.use_count() > 1 || *(found->second.isUniquePtr) == false;
                                    break;
                                }
                            }
                            if(!hasBeenFound) {
                                doCopy = true;
                            }
                        }else{ // if none of the above has been triggered, copy the data only if it has not already been duplicated
                            doCopy = true;
                            for(auto me : copyMapsToLookInto) {
                                doCopy &= me->find(h1->castPtr<TargetParamType>()) == me->end();
                            }
                            
                            mPtr = copyMapsToLookInto.back();
                        }
                        
                        if(doCopy) {
                            auto copy = this->coreCopyCreationCore(*mPtr, initialActivationState, inPriority, *h1->castPtr<TargetParamType>());
                            res[copy.originAdress] = copy;
                        }

                        indexHh += 1;
                    }

                }
            }
        , args);
        
        return res;
    }

    template <class Tuple, std::size_t N>
    inline auto copyIfMaybeWriteAndNotDuplicateOrUsedInRead(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                            const SpTaskActivation initialActivationState,
                                                            const SpPriority& inPriority,
                                                            Tuple& args){
        return copyIfAccess<false, true, SpDataAccessMode::POTENTIAL_WRITE>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }

    template <class Tuple, std::size_t N>
    inline auto copyIfMaybeWriteAndDuplicate(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                             const SpTaskActivation initialActivationState,
                                             const SpPriority& inPriority,
                                             Tuple& args){
        return copyIfAccess<true, false, SpDataAccessMode::POTENTIAL_WRITE>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }

    template <class Tuple, std::size_t N>
    inline auto copyIfWriteAndNotDuplicateOrUsedInRead(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                                                       const SpTaskActivation initialActivationState,
                                                       const SpPriority& inPriority,
                                                       Tuple& args){
        return copyIfAccess<false, true, SpDataAccessMode::WRITE>(copyMapsToLookInto, initialActivationState, inPriority, args);
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Core logic helper functions
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    void addCallbackToTask(T& inTaskView, SpGeneralSpecGroup<SpecModel>* inSg) {
        inTaskView.addCallback([this, aTaskPtr = inTaskView.getTaskPtr(), specGroupPtr = inSg]
                                (const bool alreadyDone, const bool& taskRes, SpAbstractTaskWithReturn<bool>::SpTaskViewer& /*view*/,
                                const bool isEnabled){
                                    if(isEnabled){
                                        if(!alreadyDone){
                                            specGroupMutex.lock();
                                        }
                                        
                                        specGroupPtr->setSpeculationCurrentResult(!taskRes);
                                        
                                        if(!alreadyDone){
                                            specGroupMutex.unlock();
                                        }
                                    }
                                });
    }

    template <class Tuple>
    auto getCorrespondingExecutionPaths(Tuple& args){
        
        small_vector<ExecutionPathWeakPtrTy> res;
        
        if(hashMap.empty()) {
            return res;
        }
        
        SpUtils::foreach_in_tuple(
            [this, &res](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;

                [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                long int indexHh = 0;
                for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

                    if(auto found = hashMap.find(ptr); found != hashMap.end()){
                        res.push_back(found->second);
                    }

                    indexHh += 1;
                } 
            }
        , args);
        
        auto sortLambda = [] (ExecutionPathWeakPtrTy& a, ExecutionPathWeakPtrTy& b) {
                                    return a.lock() < b.lock();
                             };
        
        auto uniqueLambda = [] (ExecutionPathWeakPtrTy& a, ExecutionPathWeakPtrTy& b) {
                                    return a.lock() == b.lock();
                             };
        
        std::sort(res.begin(), res.end(), sortLambda);
        res.erase(std::unique(res.begin(), res.end(), uniqueLambda), res.end());

        return res;
    }

    template <unsigned char flags, class Tuple>
    auto getOriginalAddressesOfHandlesWithAccessModes(Tuple& args) {
       
        small_vector<const void*> res;
        
        SpUtils::foreach_in_tuple(
            [&, this](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                
                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
                
                if constexpr((flags & (1 << static_cast<unsigned char>(accessMode))) != 0) {
                    [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                    assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                    long int indexHh = 0;
                    for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                        assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                        assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                        
                        res.push_back(ptr);
                        
                        indexHh += 1;
                    }
                }
            }
        , args);
        
        return res;
    }
    
    void setExecutionPathForOriginalAddressesInHashMap(ExecutionPathSharedPtrTy &ep, small_vector_base<const void*>& originalAddresses){
        for(auto oa : originalAddresses) {
            hashMap[oa] = ep;
        }
    }
    
    void removeOriginalAddressesFromHashMap(small_vector_base<const void*>& originalAddresses){
        for(auto oa : originalAddresses) {
            hashMap.erase(oa);
        }
    }
        
    template <class Tuple>
    bool isUsedByTask(const void* inPtr, Tuple& args) {
        bool res = false;

        SpUtils::foreach_in_tuple(
            [this, &res, inPtr](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                
                [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                [[maybe_unused]] long int indexHh = 0;
                for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                    if(ptr == inPtr) {
                        res = true;
                        break;
                    }
                }

                return res;
            }
        , args);

        return res;
    }

    template <class Tuple>
    void manageReadDuplicate(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){
        SpUtils::foreach_in_tuple(
            [&, this](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
                
                if constexpr (std::is_destructible<TargetParamType>::value && accessMode != SpDataAccessMode::READ){
                    [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                    assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                    long int indexHh = 0;
                    for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                        assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                        assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);
                        
                        if(auto found = copyMap.find(ptr); found != copyMap.end()
                                && found->second.usedInRead){
                                assert(std::is_copy_assignable<TargetParamType>::value);

                                SpCurrentCopy& cp = found->second;
                                assert(cp.latestAdress != ptr);
                                assert(cp.latestAdress);
                                if(found->second.isUniquePtr.use_count() == 1) {
                                    this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMap)}, SpTaskActivation::ENABLE, SpPriority(0),
                                                      SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                                      [](TargetParamType& output){

                                        delete &output;
                                    }).setTaskName("sp-delete");
                                }else {
                                    *(found->second.isUniquePtr) = false;
                                }
                            copyMap.erase(found);
                        }
                        
                        indexHh += 1;
                    }
                }
            }
        , args);
    }
           
    template <class Tuple>
    auto getCorrespondingCopyGroups(std::unordered_map<const void*, SpCurrentCopy>& copyMap, Tuple& args){ 
        small_vector<SpGeneralSpecGroup<SpecModel>*> res;
        
        SpUtils::foreach_in_tuple(
            [this, &copyMap, &res](auto &&scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;

                [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                long int indexHh = 0;
                for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

                    if(auto found = copyMap.find(ptr); found != copyMap.end()){
                        assert(found->second.lastestSpecGroup);
                        res.emplace_back(found->second.lastestSpecGroup);
                    }

                    indexHh += 1;
                }
            }
        , args);
       
        std::sort(res.begin(), res.end());
        res.erase(std::unique(res.begin(), res.end()), res.end());

        return res;
    }
            
    template <bool updateIsUniquePtr, class Tuple>
    void removeAllGeneric(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto, Tuple& args){
        SpUtils::foreach_in_tuple(
            [&, this](auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;
                
                if constexpr (std::is_destructible<TargetParamType>::value && accessMode != SpDataAccessMode::READ){

                    [[maybe_unused]] auto hh = this->getDataHandle(scalarOrContainerData);
                    assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                    long int indexHh = 0;
                    for(typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                        assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                        assert(hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>() == ptr);

                        if(auto found = copyMapToLookInto.find(ptr); found != copyMapToLookInto.end()){
                            assert(std::is_copy_assignable<TargetParamType>::value);
                            SpCurrentCopy& cp = found->second;
                            if(found->second.isUniquePtr.use_count() == 1) {
                                assert(cp.latestAdress);
                                #ifndef NDEBUG
                                    this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMapToLookInto)},
                                                      SpTaskActivation::ENABLE,
                                                      SpPriority(0),
                                                      SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                                      [ptr = cp.latestAdress](TargetParamType& output){
                                                            assert(ptr ==  &output);
                                                            delete &output;
                                                      }).setTaskName("sp-delete");
                                #else
                                    this->taskInternal(std::array<CopyMapPtrTy, 1>{std::addressof(copyMapToLookInto)},
                                                      SpTaskActivation::ENABLE,
                                                      SpPriority(0),
                                                      SpWrite(*reinterpret_cast<TargetParamType*>(cp.latestAdress)),
                                                      [](TargetParamType& output){
                                                            delete &output;
                                                      }).setTaskName("sp-delete");
                                #endif
                            }else {
                                if constexpr(updateIsUniquePtr) {
                                    *(found->second.isUniquePtr) = false;
                                }
                            }
                            copyMapToLookInto.erase(found);
                        }

                        indexHh += 1;
                    }
                }
            }
        , args);
    }
    
    template <class Tuple>
    void removeAllCopiesReadFrom(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto, Tuple& args){
        removeAllGeneric<false>(copyMapToLookInto, args);    
    }

    template <class Tuple>
    void removeAllCorrespondingCopies(std::unordered_map<const void*, SpCurrentCopy>& copyMapToLookInto, Tuple& args){
        removeAllGeneric<true>(copyMapToLookInto, args);
    }

    template <class Tuple, std::size_t... Is>
    static constexpr bool coreAllAreCopieableAndDeletable(std::index_sequence<Is...>){
        return ([]() {
                    using ScalarOrContainerType = std::remove_reference_t<std::tuple_element_t<Is, Tuple>>;
                    using TargetParamType = typename ScalarOrContainerType::RawHandleType;

                    return std::conjunction_v<std::is_default_constructible<TargetParamType>,
                                              std::is_copy_assignable<TargetParamType>,
                                              std::is_destructible<TargetParamType>>;
                }() && ...);
    }

    template <class Tuple>
    static constexpr bool allAreCopieableAndDeletable(){
        using TupleWithoutRef = std::remove_reference_t<Tuple>;
        return coreAllAreCopieableAndDeletable<TupleWithoutRef>(std::make_index_sequence<std::tuple_size_v<TupleWithoutRef>>{});
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Internal task creation
    ///////////////////////////////////////////////////////////////////////////

    template <template<typename...> typename TaskType, class... ParamsAndTask, std::size_t N>
    auto taskInternal(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                      const SpTaskActivation inActivation,
                      ParamsAndTask&&... inParamsAndTask){
        auto f = [&, this, inActivation] (auto isPotentialTask, auto&& priority, auto&& /* unused probability */, auto&&... partitionedParams) {
            return this->coreTaskCreationAux<TaskType, isPotentialTask>(inActivation, std::forward<decltype(priority)>(priority),
                                                                        std::forward<decltype(partitionedParams)>(partitionedParams)..., copyMapsToLookInto);
        };
        
        return this->callWithPartitionedArgs(f, std::forward<ParamsAndTask>(inParamsAndTask)...); 
    }
    
    
    template <class... ParamsAndTask, std::size_t N>
    inline auto taskInternal(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                             const SpTaskActivation inActivation,
                             ParamsAndTask&&... inParamsAndTask){
        return taskInternal<SpTask>(copyMapsToLookInto, inActivation, std::forward<ParamsAndTask>(inParamsAndTask)...);
    }
    
    template <class... ParamsAndTask, std::size_t N>
    auto taskInternalSpSelect(const std::array<CopyMapPtrTy, N>& copyMapsToLookInto,
                              bool isCarryingSurelyWrittenValuesOver,
                              const SpTaskActivation inActivation,
                              ParamsAndTask&&... inParamsAndTask){
        auto f =
        [&, this, isCarryingSurelyWrittenValuesOver, inActivation]
        (auto isPotentialTask, auto&& priority, auto&& /* unused probability */, auto&&... partitionedParams) {
            return this->coreTaskCreationAux<SpSelectTask, isPotentialTask>(inActivation, std::forward<decltype(priority)>(priority),
                                                                  std::forward<decltype(partitionedParams)>(partitionedParams)..., copyMapsToLookInto,
                                                                  isCarryingSurelyWrittenValuesOver);
        };
        
        return this->callWithPartitionedArgs(f, std::forward<ParamsAndTask>(inParamsAndTask)...); 
    }
    
    ///////////////////////////////////////////////////////////////////////////
    /// Notify function (called by scheduler when a task is ready to run)
    ///////////////////////////////////////////////////////////////////////////

    void thisTaskIsReady(SpAbstractTask* aTask, const bool isNotCalledInAContextOfTaskCreation) final {
        SpDebugPrint() << "SpTaskGraph -- thisTaskIsReady -- will test ";
        
        if(isNotCalledInAContextOfTaskCreation){
            specGroupMutex.lock();
        }
        
        SpGeneralSpecGroup<SpecModel>* specGroup = aTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>();
        
        if(specGroup && specGroup->isSpeculationNotSet()){
            if(specFormula){
                if(specFormula(this->scheduler.getNbReadyTasks(), specGroup->getAllProbability())){
                    SpDebugPrint() << "SpTaskGraph -- thisTaskIsReady -- enableSpeculation ";
                    specGroup->setSpeculationActivation(true);
                }
                else{
                    specGroup->setSpeculationActivation(false);
                }
            }
            else if(this->scheduler.getNbReadyTasks() == 0){
                SpDebugPrint() << "SpTaskGraph -- thisTaskIsReady -- enableSpeculation ";
                specGroup->setSpeculationActivation(true);
            }
            else{
                specGroup->setSpeculationActivation(false);
            }
            
            assert(!specGroup->isSpeculationNotSet());
        }
        
        if(isNotCalledInAContextOfTaskCreation){
            specGroupMutex.unlock();
        }
    }

//=====--------------------=====
//          Public API
//=====--------------------=====
public:
    ///////////////////////////////////////////////////////////////////////////
    /// Constructor
    ///////////////////////////////////////////////////////////////////////////

    explicit SpTaskGraph() : currentSpecGroup(nullptr) {
        this->scheduler.setListener(this);
    }
        
    ///////////////////////////////////////////////////////////////////////////
    /// Destructor
    ///////////////////////////////////////////////////////////////////////////

    //! Destructor waits for tasks to finish
    ~SpTaskGraph(){
        this->waitAllTasks();
        // free copies
        for(auto &e : hashMap) {
            releaseCopies(*(e.second));
        }
    }
    
    // No copy and no move
    SpTaskGraph(const SpTaskGraph&) = delete;
    SpTaskGraph(SpTaskGraph&&) = delete;
    SpTaskGraph& operator=(const SpTaskGraph&) = delete;
    SpTaskGraph& operator=(SpTaskGraph&&) = delete;

    ///////////////////////////////////////////////////////////////////////////
    /// Task creation method
    ///////////////////////////////////////////////////////////////////////////

    template <class... ParamsTy>
    auto task(ParamsTy&&...params) {
        auto f = [this](auto isPotentialTask, auto&&... partitionedParams) {
            return this->preCoreTaskCreation<isPotentialTask>(std::forward<decltype(partitionedParams)>(partitionedParams)...);
        };
        return this->callWithPartitionedArgs(f, std::forward<ParamsTy>(params)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Getters/actions
    ///////////////////////////////////////////////////////////////////////////
    
    void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula){
        specFormula = std::move(inFormula);
    }
};

template<>
class SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> : public SpTaskGraphCommon<false> {

private:
    
    template <class TaskCoreTy>
    void setDataHandlesOfTask(TaskCoreTy* aTask) {
        auto& args = aTask->getDataDependencyTupleRef();
        SpUtils::foreach_in_tuple(
            [&, this, aTask](auto index, auto&& scalarOrContainerData) {
                using ScalarOrContainerType = std::remove_reference_t<decltype(scalarOrContainerData)>;
                constexpr const SpDataAccessMode accessMode = ScalarOrContainerType::AccessMode;

                auto hh = this->getDataHandle(scalarOrContainerData);
                assert(ScalarOrContainerType::IsScalar == false || std::size(hh) == 1);
                long int indexHh = 0;
                
                for([[maybe_unused]] typename ScalarOrContainerType::HandleTypePtr ptr : scalarOrContainerData.getAllData()){
                    assert(ptr == this->getDataHandleCore(*ptr)->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    assert(ptr == hh[indexHh]->template castPtr<typename ScalarOrContainerType::RawHandleType>());
                    SpDataHandle* h = hh[indexHh];
                    
                    SpDebugPrint() << "accessMode in runtime to add dependence -- => " << SpModeToStr(accessMode);
                    
                    const long int handleKey = h->addDependence(aTask, accessMode);
                    if(indexHh == 0){
                        aTask->template setDataHandle<index>(h, handleKey);
                    }
                    else{
                        assert(ScalarOrContainerType::IsScalar == false);
                        aTask->template addDataHandleExtra<index>(h, handleKey);
                    }
                    indexHh += 1;
                }
            }
        , args);
    }
    
    template <typename CallableTy, typename TupleTy, std::size_t... Is>
    static auto apply_with_forward_impl(CallableTy&& c, TupleTy&& t, std::index_sequence<Is...>) {
        using TupleTyWithoutRef = std::remove_reference_t<TupleTy>;
        
        return std::invoke(std::forward<CallableTy>(c),
                           std::forward<std::tuple_element_t<Is, TupleTyWithoutRef>>(std::get<Is>(t))...);
    }
    
    template <typename CallableTy, typename TupleTy>
    static auto apply_with_forward(CallableTy&& c, TupleTy&& t) {
        using TupleTyWithoutRef = std::remove_reference_t<TupleTy>;
        return apply_with_forward_impl(std::forward<CallableTy>(c),
                                     std::forward<TupleTy>(t),
                                     std::make_index_sequence<std::tuple_size_v<TupleTyWithoutRef>>{});
    }
    
    template <class DataDependencyTupleTy, class CallableTupleTy>
    auto coreTaskCreation(const SpPriority& inPriority, DataDependencyTupleTy& dataDepTuple, CallableTupleTy& callableTuple) {
        SpDebugPrint() << "SpTaskGraph -- coreTaskCreation";
               
        auto createCopyTupleFunc =
        [](auto& tupleOfRefs) {
            return [&tupleOfRefs]() {
                // Here we forward to std::tuple_element_t<Is, std::remove_reference_t<DataDependencyTupleTy>> in order
                // to move the data dependency object into the new tuple when the data dependency object is an rvalue.
                // Since this is a non speculative task graph we will only insert one instance of each task and so we can safely
                // move from the data dependency object since we won't reference it again afterwards.
                return apply_with_forward([](auto&&...elt) {
                    return std::make_tuple(std::forward<decltype(elt)>(elt)...);
                }, tupleOfRefs);
            };
        };
        
        auto dataDependencyTupleCopyFunc = createCopyTupleFunc(dataDepTuple);
        auto callableTupleCopyFunc = createCopyTupleFunc(callableTuple);
                                            
        static_assert(0 < std::tuple_size<decltype(callableTupleCopyFunc())>::value );
         
        using DataDependencyTupleCopyTy = std::remove_reference_t<decltype(dataDependencyTupleCopyFunc())>;    
        using CallableTupleCopyTy = std::remove_reference_t<decltype(callableTupleCopyFunc())>;
        using RetTypeRef = decltype(apply_on_data_dep_tuple(std::get<0>(callableTuple).getCallableRef(), dataDepTuple));
        using RetType = std::remove_reference_t<RetTypeRef>;
        using TaskTy = SpTask<RetType, DataDependencyTupleCopyTy, CallableTupleCopyTy>;

        // Create a task with a copy of the callables and the data dependency objects
        auto aTask = new TaskTy(this, SpTaskActivation::ENABLE, inPriority, dataDependencyTupleCopyFunc(), callableTupleCopyFunc());

        // Lock the task
        aTask->takeControl();
        
        // Add the handles
        setDataHandlesOfTask(aTask);

        // Check coherency
        assert(aTask->areDepsCorrect());

        // The task has been initialized
        aTask->setState(SpTaskState::INITIALIZED);

        // Get the view
        auto descriptor = aTask->getViewer();

        aTask->setState(SpTaskState::WAITING_TO_BE_READY);
        
        aTask->releaseControl();

        SpDebugPrint() << "SpTaskGraph -- coreTaskCreation => " << aTask << " of id " << aTask->getId();
        
        // Push to the scheduler
        this->scheduler.addNewTask(aTask);
        
        // Return the view
        return descriptor;
    }

//=====--------------------=====
//          Public API
//=====--------------------=====
public:
    ///////////////////////////////////////////////////////////////////////////
    /// Constructor
    ///////////////////////////////////////////////////////////////////////////

    explicit SpTaskGraph() {}
        
    ///////////////////////////////////////////////////////////////////////////
    /// Destructor
    ///////////////////////////////////////////////////////////////////////////

    //! Destructor waits for tasks to finish
    ~SpTaskGraph(){
        this->waitAllTasks();
    }
    
    // No copy and no move
    SpTaskGraph(const SpTaskGraph&) = delete;
    SpTaskGraph(SpTaskGraph&&) = delete;
    SpTaskGraph& operator=(const SpTaskGraph&) = delete;
    SpTaskGraph& operator=(SpTaskGraph&&) = delete;

    ///////////////////////////////////////////////////////////////////////////
    /// Task creation method
    ///////////////////////////////////////////////////////////////////////////

    template <class... ParamsTy>
    auto task(ParamsTy&&...params) {
        auto f = [this](auto&&... partitionedParams) {
            return this->coreTaskCreation(std::forward<decltype(partitionedParams)>(partitionedParams)...);
        };
        return this->callWithPartitionedArgs(f, std::forward<ParamsTy>(params)...);
    }
};

#endif
