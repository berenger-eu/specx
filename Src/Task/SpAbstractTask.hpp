///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPABSTRACTTASK_HPP
#define SPABSTRACTTASK_HPP

#include <cassert>
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_set>

#include "Utils/SpTimePoint.hpp"
#include "Utils/SpUtils.hpp"
#include "Data/SpDataAccessMode.hpp"
#include "Task/SpPriority.hpp"
#include "Task/SpProbability.hpp"
#include "Utils/small_vector.hpp"

class SpAbstractTaskGraph;

/** The possible state of a task */
enum class SpTaskState {
    NOT_INITIALIZED, //! Is currently being build in the runtime
    INITIALIZED, //! Has been built, dependences will be checked
    WAITING_TO_BE_READY, //! Depdences are not ready
    READY, //! Dependences are ready, the task is in the "ready" list
    RUNNING, //! The task is currently being computed
    POST_RUN, //! The task is over, post dependences are checked
    FINISHED, //! The task is in the "finished" list
};

enum class SpTaskActivation{
    DISABLE,
    ENABLE
};

class SpDataHandle;

class SpSpecTaskGroup;

/**
 * This class is the interface to the real task,
 * it is used to store the id and terminaison mechanism.
 */
class SpAbstractTask{
    static std::atomic<long int> TaskIdsCounter;

    //! Current task id
    const long int taskId;
    //! Tells if a task has been executed
    std::atomic_bool hasBeenExecuted;

    //! Mutex to protected the waiting condition conditionExecuted
    mutable std::mutex mutexExecuted;
    //! Should be use to wait for task completion
    mutable std::condition_variable conditionExecuted;

    //! To protect critical part of a task
    std::mutex mutexTakeControl;
    //! Current state of the task
    std::atomic<SpTaskState> currentState;

    //! When the task has been created (construction time)
    SpTimePoint creationTime;
    //! When a task has been ready
    SpTimePoint readyTime;
    //! When the task has been computed (begin)
    SpTimePoint startingTime;
    //! When the task has been computed (end)
    SpTimePoint endingTime;

    //! The Id of the thread that computed the task
    long int threadIdComputer;

    //! Priority
    SpPriority priority;

    //! To know if the callback has to be executed
    std::atomic<SpTaskActivation> isEnabled;

    void* specTaskGroup;
    SpAbstractTask* originalTask;
    
    SpAbstractTaskGraph* const atg;

    mutable std::atomic<SpAbstractTask*> ptrNextList;

    #ifdef SPECX_COMPILE_WITH_MPI
    bool isMpiTaskCom;
#endif

protected:
    static void SetCurrentTask(SpAbstractTask* inCurrentTask);
public:
    static SpAbstractTask* GetCurrentTask();


    explicit SpAbstractTask(SpAbstractTaskGraph* const inAtg, const SpTaskActivation initialAtivationState, const SpPriority& inPriority):
        taskId(TaskIdsCounter++), hasBeenExecuted(false),
                               currentState(SpTaskState::NOT_INITIALIZED),
                               threadIdComputer(-1), priority(inPriority),
                               isEnabled(initialAtivationState),
                               specTaskGroup(nullptr),
                               originalTask(nullptr),
                               atg(inAtg),
                               ptrNextList(nullptr)
                         #ifdef SPECX_COMPILE_WITH_MPI
                             ,isMpiTaskCom(false)
    #endif
    {
    }

    virtual ~SpAbstractTask(){}

    //! A task cannot be copied or move
    SpAbstractTask(const SpAbstractTask&) = delete;
    SpAbstractTask(SpAbstractTask&&) = delete;
    SpAbstractTask& operator=(const SpAbstractTask&) = delete;
    SpAbstractTask& operator=(SpAbstractTask&&) = delete;

    ///////////////////////////////////////////////////////////////////////////
    /// Methods to be specialized by sub classes
    ///////////////////////////////////////////////////////////////////////////

    virtual long int getNbParams() = 0;
    virtual bool dependencesAreReady() const = 0;
    virtual void preTaskExecution(SpCallableType ct) = 0;
    virtual void postTaskExecution(SpAbstractTaskGraph&, SpCallableType ct) = 0;
    virtual void executeCore(SpCallableType ct) = 0;
    virtual void releaseDependences(small_vector_base<SpAbstractTask*>* potentialReady) = 0;
    virtual void getDependences(small_vector_base<SpAbstractTask*>* allDeps) const = 0;
    virtual void getPredecessors(small_vector_base<SpAbstractTask*>* allPredecessors) const = 0;
    virtual void useDependences(std::unordered_set<SpDataHandle*>* exceptionList) = 0;
    virtual bool hasMode(const SpDataAccessMode inMode) const = 0;
    virtual small_vector<std::pair<SpDataHandle*,SpDataAccessMode>> getDataHandles() const = 0;
    virtual void executeCallback() = 0;
    virtual bool hasCallableOfType(const SpCallableType sct) const = 0;
    virtual std::string getTaskBodyString() = 0;
    virtual std::string coreGetTaskName() const = 0;
    virtual void setTaskName(std::string inName) = 0;

    void useDependences() {
        useDependences(nullptr);
    }

    void execute(SpCallableType ct) {
        startingTime.setToNow();
        if(isEnabled == SpTaskActivation::ENABLE){
            executeCore(ct);
        }
        endingTime.setToNow();

        {
            std::unique_lock<std::mutex> locker(mutexExecuted);
            threadIdComputer = SpUtils::GetThreadId();
            hasBeenExecuted = true;
            conditionExecuted.notify_all();
        }
        
        executeCallback();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Getter methods
    ///////////////////////////////////////////////////////////////////////////

    int getPriority() const{
        return priority.getPriority();
    }

    const SpPriority& getPriorityObject() const{
        return priority;
    }

    long int getId() const{
        return taskId;
    }

    long int getThreadIdComputer() const{
        return threadIdComputer;
    }

    //! Will return only when the task will be over
    void wait() const {
        if(hasBeenExecuted == false){
            std::unique_lock<std::mutex> locker(mutexExecuted);
            conditionExecuted.wait(locker, [&]{ return bool(hasBeenExecuted);});
        }
    }

    SpTaskState getState() const {
        return currentState;
    }

    void setState(SpTaskState inState) {
        currentState = inState;
        if(SpTaskState::READY == currentState){
            readyTime.setToNow();
        }
    }

    bool isState(SpTaskState inState) const{
        return getState() == inState;
    }

    bool isOver() const{
        return hasBeenExecuted;
    }

    bool canTakeControl(){
        return mutexTakeControl.try_lock();
    }

    void takeControl(){
        return mutexTakeControl.lock();
    }

    void releaseControl(){
        mutexTakeControl.unlock();
    }

    //! Will return true even if the task is over
    bool isReady() const{
        return SpTaskState::READY <= getState();
    }

    // Not const ref because of the original name build on the fly
    std::string getTaskName() const{
        if(originalTask){
            return originalTask->coreGetTaskName() + "'";
        }
        else{
            return coreGetTaskName();
        }
    }

    const SpTimePoint& getCreationTime() const{
        return creationTime;
    }
    const SpTimePoint& getReadyTime() const{
        return readyTime;
    }
    const SpTimePoint& getStartingTime() const{
        return startingTime;
    }
    const SpTimePoint& getEndingTime() const{
        return endingTime;
    }

    bool isTaskEnabled() const{
        return isEnabled == SpTaskActivation::ENABLE;
    }
    
    virtual void setEnabledDelegate(const SpTaskActivation inIsEnable) {
        setEnabled(inIsEnable);
    }

    void setEnabled(const SpTaskActivation inIsEnable) {
        isEnabled = inIsEnable;
    }

    void setDisabledIfNotOver(){
        if(canTakeControl()){
            if(isOver() == false){
                isEnabled = SpTaskActivation::DISABLE;
            }
            releaseControl();
        }
    }

    template <class SpSpecGroupType>
    void setSpecGroup(SpSpecGroupType* inSpecTaskGroup){
        specTaskGroup = inSpecTaskGroup;
    }

    template <class SpSpecGroupType>
    SpSpecGroupType* getSpecGroup(){
        return reinterpret_cast<SpSpecGroupType*>(specTaskGroup);
    }

    void setOriginalTask(SpAbstractTask* inOriginal){
        originalTask = inOriginal;
    }

    bool isEnable() const {
        return isEnabled == SpTaskActivation::ENABLE;
    }
    
    SpAbstractTaskGraph* getAbstractTaskGraph() const {
        return atg;
    }

#ifdef SPECX_COMPILE_WITH_MPI
    bool isMpiCom() const{
        return isMpiTaskCom;
    }
    void setIsMpiCom(const bool inIsMpiCom){
        isMpiTaskCom = inIsMpiCom;
    }
#endif


    std::atomic<SpAbstractTask*>& getPtrNextList(){
        return ptrNextList;
    }
    std::atomic<SpAbstractTask*>& getPtrNextList() const {
        return ptrNextList;
    }
};


///////////////////////////////////////////////////////////////////////////
/// Specialization for tasks that return a value
///////////////////////////////////////////////////////////////////////////


#include <future>
#include <functional>

//! If the return type is not void
template <class RetType>
class SpAbstractTaskWithReturn : public SpAbstractTask {
public:

    class SpTaskViewer{
        SpAbstractTaskWithReturn<RetType>* target;
        
        SpTaskViewer(SpAbstractTaskWithReturn<RetType>* inTarget) : target(inTarget){}
    public:
        SpTaskViewer() : target(nullptr) {}
        SpTaskViewer(const SpTaskViewer&) = default;
        SpTaskViewer(SpTaskViewer&&) = default;
        SpTaskViewer& operator=(const SpTaskViewer&) = default;
        SpTaskViewer& operator=(SpTaskViewer&&) = default;

        void wait() const {
            target->wait();
        }

        bool isOver() const{
            return target->isOver();
        }

        bool isReady() const{
            return target->isReady();
        }

        RetType getValue() const{
            return target->getValue();
        }

        // Not const ref because of the original name build on the fly
        std::string getTaskName() const{
            return target->getTaskName();
        }

        void setTaskName(const std::string& inTaskName){
            target->setTaskName(inTaskName);
        }

        SpAbstractTask* getTaskPtr(){
            return target;
        }

        void addCallback(std::function<void(const bool, const RetType&, SpTaskViewer&, const bool)> func){
            target->addCallback(std::move(func));
        }

        void setOriginalTask(SpAbstractTask* inOriginal){// For speculation
            target->setOriginalTask(inOriginal);
        }

        bool hasCallableOfType(const SpCallableType sct) const {
            return target->hasCallableOfType(sct);
        }

        friend SpAbstractTaskWithReturn;
    };
private:
    RetType resultValue;

    std::mutex mutexCallbacks;
    small_vector<std::function<void(const bool, const RetType&, SpTaskViewer&, const bool)>> callbacks;

public:
    explicit SpAbstractTaskWithReturn(SpAbstractTaskGraph* const inAtg, const SpTaskActivation initialAtivationState, const SpPriority& inPriority):
        SpAbstractTask(inAtg, initialAtivationState, inPriority), resultValue(){
    }

    const RetType& getValue() const {
        wait();
        return resultValue;
    }

    void setValue(RetType&& inValue){
        resultValue = std::move(inValue);
    }

    void executeCallback() final{
        std::unique_lock<std::mutex> locker(mutexCallbacks);
        SpTaskViewer viewer(this);
        for(auto& func : callbacks){
            func(false, resultValue, viewer, isEnable());
        }
    }

    void addCallback(std::function<void(const bool, const RetType&, SpTaskViewer&, const bool)> func){
        std::unique_lock<std::mutex> locker(mutexCallbacks);
        if(isOver() == false){
            callbacks.emplace_back(std::move(func));
        }
        else{
            SpTaskViewer viewer(this);
            func(true, resultValue,viewer, isEnable());
        }
    }


    SpTaskViewer getViewer() {
        return SpTaskViewer(this);
    }
};

//! If the return type is void
template <>
class SpAbstractTaskWithReturn<void> : public SpAbstractTask {
public:

    class SpTaskViewer{
        SpAbstractTaskWithReturn<void>* target;

        SpTaskViewer(SpAbstractTaskWithReturn<void>* inTarget) : target(inTarget){}
    public:
        SpTaskViewer() : target(nullptr) {}
        SpTaskViewer(const SpTaskViewer&) = default;
        SpTaskViewer(SpTaskViewer&&) = default;
        SpTaskViewer& operator=(const SpTaskViewer&) = default;
        SpTaskViewer& operator=(SpTaskViewer&&) = default;

        void wait() const {
            target->wait();
        }

        bool isOver() const{
            return target->isOver();
        }

        bool isReady() const{
            return target->isReady();
        }

        void getValue() {
            wait();
        }

        // Not const ref because of the original name build on the fly
        std::string getTaskName() const{
            return target->getTaskName();
        }

        void setTaskName(const std::string& inTaskName){
            target->setTaskName(inTaskName);
        }

        SpAbstractTask* getTaskPtr(){
            return target;
        }

        void addCallback(std::function<void(const bool, SpTaskViewer&, const bool)> func){
            target->addCallback(std::move(func));
        }


        void setOriginalTask(SpAbstractTask* inOriginal){// For speculation
            target->setOriginalTask(inOriginal);
        }
        
        bool hasCallableOfType(const SpCallableType sct) const {
            return target->hasCallableOfType(sct);
        }

        friend SpAbstractTaskWithReturn;
    };
private:
    std::mutex mutexCallbacks;
    small_vector<std::function<void(const bool, SpTaskViewer&, const bool)>> callbacks;

public:
    using SpAbstractTask::SpAbstractTask;

    void setValue(void) = delete;

    void executeCallback() final{
        std::unique_lock<std::mutex> locker(mutexCallbacks);
        SpTaskViewer viewer(this);
        for(auto& func : callbacks){
            func(false, viewer, isEnable());
        }
    }

    void addCallback(std::function<void(const bool, SpTaskViewer&, const bool)> func){
        std::unique_lock<std::mutex> locker(mutexCallbacks);
        if(isOver() == false){
            callbacks.emplace_back(std::move(func));
        }
        else{
            SpTaskViewer viewer(this);
            func(true,viewer, isEnable());
        }
    }

    SpTaskViewer getViewer() {
        return SpTaskViewer(this);
    }
};


#endif
