///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPTASKSMANAGER_HPP
#define SPTASKSMANAGER_HPP

#include <functional>
#include <list>
#include <unordered_map>
#include <memory>
#include <unistd.h>
#include <fstream>
#include <cmath>
#include <atomic>

#include "Utils/SpUtils.hpp"
#include "Tasks/SpAbstractTask.hpp"
#include "Utils/SpPriority.hpp"
#include "Utils/SpTimePoint.hpp"
#include "SpSimpleScheduler.hpp"
#include "SpPrioScheduler.hpp"
#include "SpSchedulerInformer.hpp"
#include "Utils/small_vector.hpp"
#include "Compute/SpComputeEngine.hpp"

//! The runtime is the main component of spetabaru.
class SpTasksManager{

    std::atomic<SpComputeEngine*> ce;

    //! To protect tasksFinished
    std::mutex mutexFinishedTasks;
    
    //! List of tasks finished
    std::list<SpAbstractTask*> tasksFinished;
    
    //! To wait all tasks to be over
    std::condition_variable conditionAllTasksOver;

    //! Number of currently running tasks
    std::atomic<int> nbRunningTasks;
    
    //! Number of added tasks
    std::atomic<int> nbPushedTasks;
    
    //! Number of tasks that are ready
    std::atomic<int> nbReadyTasks;

    //! To protect commute locking
    std::mutex mutexCommute;
    
    small_vector<SpAbstractTask*> readyTasks;

    void insertIfReady(SpAbstractTask* aTask){
        if(aTask->isState(SpTaskState::WAITING_TO_BE_READY)){
            aTask->takeControl();
            if(aTask->isState(SpTaskState::WAITING_TO_BE_READY)){
                SpDebugPrint() << "Is waiting to be ready " << aTask->getId();
                const bool useCommute = aTask->hasMode(SpDataAccessMode::COMMUTE_WRITE);
                if(useCommute){
                    mutexCommute.lock();
                }
                if(aTask->dependencesAreReady()){
                    aTask->useDependences();
                    if(useCommute){
                        mutexCommute.unlock();
                    }
                    SpDebugPrint() << "Was not in ready list " << aTask->getId();

                    nbReadyTasks++;

                    aTask->setState(SpTaskState::READY);
                    aTask->releaseControl();
                    informAllReady(aTask);
                    
                    if(!ce) {
                        readyTasks.push_back(aTask);
                    } else {
                        ce.load()->pushTask(aTask);
                    }
                }
                else{
                    SpDebugPrint() << " not ready yet " << aTask->getId();
                    aTask->releaseControl();
                    if(useCommute){
                       mutexCommute.unlock();
                    }
                }
            }
            else{
                aTask->releaseControl();
            }
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    
    std::mutex listenersReadyMutex;
    small_vector<SpAbstractToKnowReady*> listenersReady;
    
    void informAllReady(SpAbstractTask* aTask){
        if(lockerByThread0 == false || SpUtils::GetThreadId() != 0){
            listenersReadyMutex.lock();
        }
        for(SpAbstractToKnowReady* listener : listenersReady){
            listener->thisTaskIsReady(aTask);
        }
        if(lockerByThread0 == false || SpUtils::GetThreadId() != 0){
            listenersReadyMutex.unlock();
        }
    }

    std::atomic<bool> lockerByThread0;

public:
    void lockListenersReadyMutex(){
        assert(lockerByThread0 == false);
        assert(SpUtils::GetThreadId() == 0);
        lockerByThread0 = true;
        listenersReadyMutex.lock();
    }

    void unlockListenersReadyMutex(){
        assert(lockerByThread0 == true);
        assert(SpUtils::GetThreadId() == 0);
        lockerByThread0 = false;
        listenersReadyMutex.unlock();
    }


    void registerListener(SpAbstractToKnowReady* aListener){
        std::unique_lock<std::mutex> locker(listenersReadyMutex);
        listenersReady.push_back(aListener);
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    explicit SpTasksManager() : ce(nullptr), nbRunningTasks(0), nbPushedTasks(0), nbReadyTasks(0),
        lockerByThread0(false){
    }

    // No copy or move
    SpTasksManager(const SpTasksManager&) = delete;
    SpTasksManager(SpTasksManager&&) = delete;
    SpTasksManager& operator=(const SpTasksManager&) = delete;
    SpTasksManager& operator=(SpTasksManager&&) = delete;

    ~SpTasksManager(){
        waitAllTasks();
        
        // Delete tasks
        for(auto ptrTask : tasksFinished){
            delete ptrTask;
        }
    }

    const std::list<SpAbstractTask*>& getFinishedTaskList() const{
        return tasksFinished;
    }

    int getNbRunningTasks() const{
        return nbRunningTasks;
    }

    //! Wait all tasks to be finished
    void waitAllTasks(){
        {
            std::unique_lock<std::mutex> locker(mutexFinishedTasks);
            SpDebugPrint() << "Waiting for  " << tasksFinished.size() << " to finish over " << nbPushedTasks << " created tasks";
            conditionAllTasksOver.wait(locker, [&](){return static_cast<long int>(tasksFinished.size()) == nbPushedTasks;});
        }
        
        assert(nbRunningTasks == 0);
    }

    //! Wait until windowSize or less tasks are pending
    void waitRemain(const long int windowSize){
        std::unique_lock<std::mutex> locker(mutexFinishedTasks);
        SpDebugPrint() << "Waiting for  " << tasksFinished.size() << " to finish over " << nbPushedTasks << " created tasks";
        conditionAllTasksOver.wait(locker, [&](){return nbPushedTasks - static_cast<long int>(tasksFinished.size()) <= windowSize;});
    }


    void addNewTask(SpAbstractTask* newTask){
        nbPushedTasks += 1;
        insertIfReady(newTask);
    }

    int getNbReadyTasks() const{
        return nbReadyTasks;
    }
    
    void setComputeEngine(SpComputeEngine* inCe) {
        if(inCe && !ce) {
            ce = inCe;
            ce.load()->pushTasks(readyTasks);
            readyTasks.clear();
        }
    }
    
    void preTaskExecution(SpAbstractTask* t) {
        nbReadyTasks--;
        nbRunningTasks += 1;
        t->takeControl();

        SpDebugPrint() << "Execute task with ID " << t->getId();
        assert(t->isState(SpTaskState::READY));

        t->setState(SpTaskState::RUNNING);
    }

    void postTaskExecution(SpAbstractTask* t){
        t->setState(SpTaskState::POST_RUN);
        
        small_vector<SpAbstractTask*> candidates;
        t->releaseDependences(&candidates);

        SpDebugPrint() << "Proceed candidates from after " << t->getId() << ", they are " << candidates.size();
        for(auto otherId : candidates){
            SpDebugPrint() << "Test " << otherId->getId();
            insertIfReady(otherId);
        }

        {
            std::unique_lock<std::mutex> locker(mutexFinishedTasks);
            tasksFinished.emplace_back(t);
            nbRunningTasks -= 1;
        }
        
        t->setState(SpTaskState::FINISHED);
        t->releaseControl();
        
        std::unique_lock<std::mutex> locker(mutexFinishedTasks);
        conditionAllTasksOver.notify_one();
    }
    
    const SpComputeEngine* getComputeEngine() const {
        return ce;
    }
};


#endif
