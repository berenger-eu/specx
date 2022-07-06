///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPTASKMANAGER_HPP
#define SPTASKMANAGER_HPP

#include <functional>
#include <list>
#include <unordered_map>
#include <memory>
#include <unistd.h>
#include <fstream>
#include <cmath>
#include <atomic>

#include "Utils/SpUtils.hpp"
#include "Task/SpAbstractTask.hpp"
#include "Task/SpPriority.hpp"
#include "Utils/SpTimePoint.hpp"
#include "SpSimpleScheduler.hpp"
#include "SpPrioScheduler.hpp"
#include "Utils/small_vector.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorker.hpp"
#include "SpTaskManagerListener.hpp"

class SpAbstractTaskGraph;

//! The runtime is the main component of specx.
class SpTaskManager{

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
    
    //! Number of finished tasks
    std::atomic<int> nbFinishedTasks;

    //! To protect commutative data accesses
    std::mutex mutexCommutative;
    
    small_vector<SpAbstractTask*> readyTasks;

    template <const bool isNotCalledInAContextOfTaskCreation>
    void insertIfReady(SpAbstractTask* aTask){
        if(aTask->isState(SpTaskState::WAITING_TO_BE_READY)){
            aTask->takeControl();
            if(aTask->isState(SpTaskState::WAITING_TO_BE_READY)){
                SpDebugPrint() << "[insertIfReady] Is waiting to be ready, id " << aTask->getId();
                const bool hasCommutativeAccessMode = aTask->hasMode(SpDataAccessMode::COMMUTATIVE_WRITE);
                if(hasCommutativeAccessMode){
                    mutexCommutative.lock();
                }
                if(aTask->dependencesAreReady()){
                    aTask->useDependences();
                    if(hasCommutativeAccessMode){
                        mutexCommutative.unlock();
                    }
                    SpDebugPrint() << "[insertIfReady] Was not in ready list, id " << aTask->getId();

                    nbReadyTasks++;

                    aTask->setState(SpTaskState::READY);
                    aTask->releaseControl();

#ifdef SPECX_COMPILE_WITH_MPI
                    if(aTask->isMpiCom()){
                        SpDebugPrint() << "[insertIfReady] is mpi task " << aTask->getId();
                        this->preMPITaskExecution(aTask);
                        aTask->executeCore(SpCallableType::CPU);
                    }
                    else{
                        SpDebugPrint() << "[insertIfReady] is normal task " << aTask->getId();
#endif
                        auto l = listener.load();

                        if(l) {
                            l->thisTaskIsReady(aTask, isNotCalledInAContextOfTaskCreation);
                        }

                        if(!ce) {
                            readyTasks.push_back(aTask);
                        } else {
                            ce.load()->pushTask(aTask);
                        }
#ifdef SPECX_COMPILE_WITH_MPI
                    }
#endif
                }
                else{
                    SpDebugPrint() << "[insertIfReady] not ready yet, id " << aTask->getId();
                    aTask->releaseControl();
                    if(hasCommutativeAccessMode){
                       mutexCommutative.unlock();
                    }
                }
            }
            else{
                aTask->releaseControl();
            }
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////
    std::atomic<SpTaskManagerListener*> listener;
    
public:
    
    void setListener(SpTaskManagerListener* inListener){
        listener = inListener;
    }

    ///////////////////////////////////////////////////////////////////////////////////////

    explicit SpTaskManager() : ce(nullptr), nbRunningTasks(0), nbPushedTasks(0), nbReadyTasks(0), nbFinishedTasks(0), listener(nullptr) {}

    // No copy or move
    SpTaskManager(const SpTaskManager&) = delete;
    SpTaskManager(SpTaskManager&&) = delete;
    SpTaskManager& operator=(const SpTaskManager&) = delete;
    SpTaskManager& operator=(SpTaskManager&&) = delete;

    ~SpTaskManager(){
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
        nbPushedTasks++;
        insertIfReady<false>(newTask);
    }

    int getNbReadyTasks() const {
        return nbReadyTasks;
    }
    
    void preTaskExecution(SpAbstractTaskGraph& atg, SpAbstractTask* t, SpWorker& w);
    void postTaskExecution(SpAbstractTaskGraph& atg, SpAbstractTask* t, SpWorker& w);
#ifdef SPECX_COMPILE_WITH_MPI
    void preMPITaskExecution(SpAbstractTask* t);
    void postMPITaskExecution(SpAbstractTaskGraph& atg, SpAbstractTask* t);
#endif
    
    void setComputeEngine(SpComputeEngine* inCe) {
        if(inCe && !ce) {
            ce = inCe;
            ce.load()->pushTasks(readyTasks);
            readyTasks.clear();
        }
    }
    
    const SpComputeEngine* getComputeEngine() const {
        return ce;
    }
    
    bool isFinished() const {
        return nbFinishedTasks == nbPushedTasks;
    }
};

#include "TaskGraph/SpAbstractTaskGraph.hpp"

inline void SpTaskManager::preTaskExecution([[maybe_unused]] SpAbstractTaskGraph& atg, SpAbstractTask* t, SpWorker& w) {
    SpDebugPrint() << "[preTaskExecution] task " << t->getId();
	nbReadyTasks--;
	nbRunningTasks += 1;
	t->takeControl();
	
	if constexpr(SpConfig::CompileWithCuda) {	
		switch(w.getType()) {
			case SpWorker::SpWorkerType::CPU_WORKER:
                t->preTaskExecution(SpCallableType::CPU);
                break;
#ifdef SPECX_COMPILE_WITH_CUDA
			case SpWorker::SpWorkerType::CUDA_WORKER:
                t->preTaskExecution(SpCallableType::CUDA);
				break;
#endif
			default:
				assert(false && "Worker is of unknown type.");
		}
	}

	SpDebugPrint() << "Execute task with ID " << t->getId();
	assert(t->isState(SpTaskState::READY));

	t->setState(SpTaskState::RUNNING);
}

#ifdef SPECX_COMPILE_WITH_MPI
inline void SpTaskManager::preMPITaskExecution(SpAbstractTask* t) {
    SpDebugPrint() << "[preMPITaskExecution] task " << t->getId();
    nbReadyTasks--;
    nbRunningTasks += 1;
    t->takeControl();

    t->preTaskExecution(SpCallableType::CPU);

    SpDebugPrint() << "Execute task with ID " << t->getId();
    assert(t->isState(SpTaskState::READY));

    t->setState(SpTaskState::RUNNING);
}
#endif


inline void SpTaskManager::postTaskExecution(SpAbstractTaskGraph& atg, SpAbstractTask* t, SpWorker& w) {
	t->setState(SpTaskState::POST_RUN);
	
	if constexpr(SpConfig::CompileWithCuda) {	
		switch(w.getType()) {
			case SpWorker::SpWorkerType::CPU_WORKER:
				t->postTaskExecution(atg, SpCallableType::CPU);
				break;
                #ifdef SPECX_COMPILE_WITH_CUDA
			case SpWorker::SpWorkerType::CUDA_WORKER:
				t->postTaskExecution(atg, SpCallableType::CUDA);
				break;
#endif
			default:
				assert(false && "Worker is of unknown type.");
		}
	}
	
	small_vector<SpAbstractTask*> candidates;
	t->releaseDependences(&candidates);

    SpDebugPrint() << "[postTaskExecution] Proceed candidates from after " << t->getId() << ", they are " << candidates.size();
	for(auto otherId : candidates){
        SpDebugPrint() << "[postTaskExecution] Test task id " << otherId->getId();
		insertIfReady<true>(otherId);
	}
    SpDebugPrint() << "[postTaskExecution] nbReadyTasks is now " << nbReadyTasks;
	
	t->setState(SpTaskState::FINISHED);
	t->releaseControl();
	
	nbRunningTasks--;
	
	// We save all of the following values because the SpTaskManager
	// instance might get destroyed as soon as the mutex (mutexFinishedTasks)
	// protected region below has been executed.
	auto previousCntVal = nbFinishedTasks.fetch_add(1);
	auto nbPushedTasksVal = nbPushedTasks.load();
	SpComputeEngine *saveCe = ce.load();
		
	{
		// In this case the lock on mutexFinishedTasks should be held
		// while doing the notify on conditionAllTasksOver
		// (conditionAllTasksOver.notify_one()) because we don't want
		// the condition variable to get destroyed before we were able
		// to notify.
		std::unique_lock<std::mutex> locker(mutexFinishedTasks);
		tasksFinished.emplace_back(t);
		
		// We notify conditionAllTasksOver every time because of
		// waitRemain  
		conditionAllTasksOver.notify_one();
	}
	
	if(nbPushedTasksVal == (previousCntVal + 1)) {
		saveCe->wakeUpWaitingWorkers();
	}
}

#ifdef SPECX_COMPILE_WITH_MPI
inline void SpTaskManager::postMPITaskExecution(SpAbstractTaskGraph& atg, SpAbstractTask* t) {
    t->setState(SpTaskState::POST_RUN);

    t->postTaskExecution(atg, SpCallableType::CPU);

    small_vector<SpAbstractTask*> candidates;
    t->releaseDependences(&candidates);

    SpDebugPrint() << "[postMPITaskExecution] Proceed candidates from after MPI " << t->getId() << ", they are " << candidates.size();
    for(auto otherId : candidates){
        SpDebugPrint() << "Test " << otherId->getId();
        insertIfReady<true>(otherId);
    }
    SpDebugPrint() << "[postMPITaskExecution] nbReadyTasks is now " << nbReadyTasks;

    t->setState(SpTaskState::FINISHED);
    t->releaseControl();

    nbRunningTasks--;

    // We save all of the following values because the SpTaskManager
    // instance might get destroyed as soon as the mutex (mutexFinishedTasks)
    // protected region below has been executed.
    auto previousCntVal = nbFinishedTasks.fetch_add(1);
    auto nbPushedTasksVal = nbPushedTasks.load();
    SpComputeEngine *saveCe = ce.load();

    {
        // In this case the lock on mutexFinishedTasks should be held
        // while doing the notify on conditionAllTasksOver
        // (conditionAllTasksOver.notify_one()) because we don't want
        // the condition variable to get destroyed before we were able
        // to notify.
        std::unique_lock<std::mutex> locker(mutexFinishedTasks);
        tasksFinished.emplace_back(t);

        SpDebugPrint() << "[postMPITaskExecution] => task "  << t->getId() << " tasksFinished.size " << tasksFinished.size() << " nbPushedTasks " << nbPushedTasks;
        // We notify conditionAllTasksOver every time because of
        // waitRemain
        conditionAllTasksOver.notify_one();
    }

    if(nbPushedTasksVal == (previousCntVal + 1)) {
        saveCe->wakeUpWaitingWorkers();
    }
    SpDebugPrint() << "[postMPITaskExecution] done";
}
#endif

#endif
