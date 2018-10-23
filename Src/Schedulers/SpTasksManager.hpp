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

#include "Utils/SpUtils.hpp"
#include "Tasks/SpAbstractTask.hpp"
#include "Utils/SpPriority.hpp"
#include "Utils/SpTimePoint.hpp"
#include "SpSimpleScheduler.hpp"
#include "SpPrioScheduler.hpp"
#include "SpSchedulerInformer.hpp"

//! The runtime is the main component of spetabaru.
class SpTasksManager{
    //! To stop the threads
    std::atomic<bool> stop;

    //! To protect the tasksReady list
    std::mutex mutexWaitingWorkers;

    //! To protect tasksFinished
    std::mutex mutexFinishedTasks;
    //! List of tasks finished
    std::list<SpAbstractTask*> tasksFinished;

    //! To wait some tasks to be ready
    std::condition_variable conditionReadyTasks;

    //! Number of currently running tasks
    std::atomic<int> nbRunningTasks;

    //! To wait all tasks to be over
    std::condition_variable conditionAllTasksOver;

    //! Number of waiting threads
    std::atomic<int> nbWaitingThreads;

    //! To protect commute locking
    std::mutex mutexCommute;

    //! Number of added tasks
    std::atomic<int> nbPushedTasks;

    //! The scheduler
    //SpSimpleScheduler scheduler;
    SpPrioScheduler scheduler;

    void insertIfReady(SpAbstractTask* aTask){
        if(aTask->isState(SpTaskState::WAITING_TO_BE_READY) && aTask->canTakeControl()){
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

                    aTask->setState(SpTaskState::READY);
                    aTask->releaseControl();
                    informAllReady(aTask);
                    const int nbThreadsToAwake = scheduler.push(aTask);

                    if(nbThreadsToAwake && nbWaitingThreads){
                        std::unique_lock<std::mutex> lockerWaiting(mutexWaitingWorkers);
                        SpDebugPrint() << "Notify other " << aTask->getId() << " nbThreadsToAwake " << nbThreadsToAwake;
                        if(nbThreadsToAwake == 1){
                            conditionReadyTasks.notify_one();
                        }
                        else{
                            conditionReadyTasks.notify_all();
                        }
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
    std::vector<SpAbstractToKnowReady*> listenersReady;
    
    void informAllReady(SpAbstractTask* aTask){
        std::unique_lock<std::mutex> locker(listenersReadyMutex);
        for(SpAbstractToKnowReady* listener : listenersReady){
            listener->thisTaskIsReady(aTask);
        }
    }

public:
    void registerListener(SpAbstractToKnowReady* aListener){
        std::unique_lock<std::mutex> locker(listenersReadyMutex);
        listenersReady.push_back(aListener);
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    explicit SpTasksManager() : stop(false), nbRunningTasks(0), nbWaitingThreads(0), nbPushedTasks(0){
    }

    // No copy or move
    SpTasksManager(const SpTasksManager&) = delete;
    SpTasksManager(SpTasksManager&&) = delete;
    SpTasksManager& operator=(const SpTasksManager&) = delete;
    SpTasksManager& operator=(SpTasksManager&&) = delete;

    ~SpTasksManager(){
        waitAllTasks();
        stopAllWorkers();

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

    //! Ask the workers to stop
    //! There must be not pending tasks
    void stopAllWorkers(){
        assert(nbRunningTasks == 0);
#ifndef NDEBUG
        {
            std::unique_lock<std::mutex> locker(mutexFinishedTasks);
            assert(static_cast<long int>(tasksFinished.size()) == nbPushedTasks);
        }
#endif

        stop = true;
        {
            std::unique_lock<std::mutex> lockerWaiting(mutexWaitingWorkers);
            conditionReadyTasks.notify_all();
        }
        assert(scheduler.getNbTasks() == 0);
    }

    //! Wait all tasks to be finished
    void waitAllTasks(){
        {
            std::unique_lock<std::mutex> locker(mutexFinishedTasks);
            SpDebugPrint() << "Waiting for  " << tasksFinished.size() << " to finish over " << nbPushedTasks << " created tasks";
            conditionAllTasksOver.wait(locker, [&](){return static_cast<long int>(tasksFinished.size()) == nbPushedTasks;});
        }

        assert(scheduler.getNbTasks() == 0);
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
        return scheduler.getNbTasks();
    }

    void runnerCallback(){
        while(stop == false){
            if(scheduler.getNbTasks() == 0){
                std::unique_lock<std::mutex> lockerWaiting(mutexWaitingWorkers);
                nbWaitingThreads += 1;
                SpDebugPrint() << "Wait for tasks";
                conditionReadyTasks.wait(lockerWaiting,
                                                       [&]{return scheduler.getNbTasks() != 0 || stop;});
                nbWaitingThreads -= 1;
            }

            SpDebugPrint() << "Awake";

            SpAbstractTask* taskToManage = nullptr;
            if(stop == false && scheduler.getNbTasks() != 0 && (taskToManage = scheduler.pop())){
                assert(taskToManage->isState(SpTaskState::READY));
                SpDebugPrint() << "There are " << scheduler.getNbTasks() << " tasks left but I have one";
                nbRunningTasks += 1;
                taskToManage->takeControl();

                SpDebugPrint() << "Execute task with ID " << taskToManage->getId();
                assert(taskToManage->isState(SpTaskState::READY));

                taskToManage->setState(SpTaskState::RUNNING);

                taskToManage->execute();

                taskToManage->setState(SpTaskState::POST_RUN);

                std::vector<SpAbstractTask*> candidates;
                taskToManage->releaseDependences(&candidates);

                SpDebugPrint() << "Proceed candidates from after " << taskToManage->getId() << ", they are " << candidates.size();
                for(auto otherId : candidates){
                    SpDebugPrint() << "Test " << otherId->getId();
                    insertIfReady(otherId);
                }

                {
                    std::unique_lock<std::mutex> locker(mutexFinishedTasks);
                    tasksFinished.emplace_back(taskToManage);
                    nbRunningTasks -= 1;
                }

                taskToManage->setState(SpTaskState::FINISHED);
                taskToManage->releaseControl();

                std::unique_lock<std::mutex> locker(mutexFinishedTasks);
                conditionAllTasksOver.notify_one();
            }
        }
    }
};


#endif
