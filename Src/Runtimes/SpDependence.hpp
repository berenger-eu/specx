///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDEPENDENCE_HPP
#define SPDEPENDENCE_HPP


#include <vector>
#include <cassert>
#include <algorithm>

#include "Utils/SpModes.hpp"
#include "Utils/small_vector.hpp"

// Reference
class SpAbstractTask;

//! This is the relation between the data and the task
class SpDependence{
    //! The access mode of the tasks on the data
    SpDataAccessMode accessMode;

    //! Id of task if it is a write mode
    SpAbstractTask* idTaskWrite;
    //! Ids of all tasks for all other modes (that are concurent safe)
    small_vector<SpAbstractTask*> idTasksMultiple;

    //! Number of tasks that use and have used the data
    long int nbTasksInUsed;
    //! Number of tasks that release the data
    long int nbTasksReleased;

public:
    explicit SpDependence(SpAbstractTask* inFirstTaskId, const SpDataAccessMode inMode)
        : accessMode(inMode) ,idTaskWrite(nullptr),  nbTasksInUsed(0) , nbTasksReleased(0){
        SpDebugPrint() << "[SpDependence] => " << inFirstTaskId << " mode " << SpModeToStr(inMode);
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            idTaskWrite = inFirstTaskId;
        }
        else{
            idTasksMultiple.push_back(inFirstTaskId);
        }
    }

    //! Can be copied and moved
    SpDependence(const SpDependence&) = default;
    SpDependence(SpDependence&&) = default;
    SpDependence& operator=(const SpDependence&) = default;
    SpDependence& operator=(SpDependence&&) = default;

    //! The access mode
    SpDataAccessMode getMode() const {
        return accessMode;
    }

    //! Add a task to the list of users
    void addTaskForMultiple(SpAbstractTask* inOtherTaskId){
        assert(accessMode != SpDataAccessMode::WRITE && accessMode != SpDataAccessMode::POTENTIAL_WRITE);
        idTasksMultiple.push_back(inOtherTaskId);
    }

    //! To know if a task can use the current data dep
    //! the task must be valid and registered
    bool canBeUsedByTask(const SpAbstractTask* useTaskId) const {
        SpDebugPrint() << "[SpDependence]canBeUsedByTask " << useTaskId;
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            // If it is write
            SpDebugPrint() << "[SpDependence]Is in write, next task is " << idTaskWrite;
            // Must not have been already used
            assert(nbTasksInUsed == 0);
            // The task id must be the one register
            assert(idTaskWrite == useTaskId);
            return true;
        }
        // If it is commute
        else if(accessMode == SpDataAccessMode::COMMUTATIVE_WRITE){
            // The number of already user tasks must be less than the number of register tasks
            assert(nbTasksInUsed < static_cast<long int>(idTasksMultiple.size()));
            // In commute their must be only 0 or 1 usage at a time
            assert(nbTasksInUsed-nbTasksReleased >= 0 && nbTasksInUsed-nbTasksReleased <= 1);
            // The given task must exist in the list
            assert(std::find(idTasksMultiple.begin(), idTasksMultiple.end(), useTaskId) != idTasksMultiple.end());
            SpDebugPrint() << "[SpDependence]Is in commute, test existence among " << idTasksMultiple.size() << " tasks";
            SpDebugPrint() << "[SpDependence]Found " << (std::find(idTasksMultiple.begin(), idTasksMultiple.end(), useTaskId) != idTasksMultiple.end());
            SpDebugPrint() << "[SpDependence]nbTasksInUsed " << nbTasksInUsed << " nbTasksReleased " << nbTasksReleased;
            // Return true if no task uses the data
            return nbTasksInUsed-nbTasksReleased == 0;
        }
        // If it is not write and not commute
        else {
            // The number of already user tasks must be less than the number of register tasks
            assert(nbTasksInUsed < static_cast<long int>(idTasksMultiple.size()));
            // The given task must exist in the list
            assert(std::find(idTasksMultiple.begin(), idTasksMultiple.end(), useTaskId) != idTasksMultiple.end());
            SpDebugPrint() << "[SpDependence]Is not in commute and not in write, test existence among " << idTasksMultiple.size() << " tasks";
            SpDebugPrint() << "[SpDependence]Found " << (std::find(idTasksMultiple.begin(), idTasksMultiple.end(), useTaskId) != idTasksMultiple.end());
            return true;
        }
    }

    //! Mark the dependence as used
    void setUsedByTask([[maybe_unused]] SpAbstractTask* useTaskId){
        assert(canBeUsedByTask(useTaskId) == true);
        nbTasksInUsed += 1;
    }
    
    void fillWithTaskList(small_vector_base<SpAbstractTask*>* potentialReady) const {
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            potentialReady->push_back(idTaskWrite);
        }
        else{
            potentialReady->reserve(potentialReady->size() + idTasksMultiple.size());
            for(auto&& ptr : idTasksMultiple){
                potentialReady->push_back(ptr);
            }
        }
    }

    //! Copy all the tasks related to the dependence into the given vector
    void fillWithListOfPotentiallyReadyTasks(small_vector_base<SpAbstractTask*>* potentialReady) const {
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            if(idTaskWrite->isState(SpTaskState::WAITING_TO_BE_READY)) {
                potentialReady->push_back(idTaskWrite);
            }
        }
        else{
            potentialReady->reserve(potentialReady->size() + idTasksMultiple.size());
            for(auto&& ptr : idTasksMultiple){
                if(ptr->isState(SpTaskState::WAITING_TO_BE_READY)) {
                    potentialReady->push_back(ptr);
                }
            }
        }
    }

    //! Marks the dependence as release by the given task
    //! Must be called after setUsedByTask
    bool releaseByTask([[maybe_unused]] SpAbstractTask* useTaskId){
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            assert(nbTasksReleased == 0);
            assert(nbTasksInUsed == 1);
            assert(idTaskWrite == useTaskId);
            nbTasksReleased += 1;
            // The dependency slot can only be used by the task currently releasing the
            // the dependency slot, since write and maybe-write accesses are exclusive.
            // After this dependency slot has been released it can't be reused by another task, 
            // that's why we are returning false in this case, basically saying this dependency slot 
            // is not available for any further memory access requests and we should move onto the next dependency slot.
            return false;
        }
        else{
            assert(std::find(idTasksMultiple.begin(), idTasksMultiple.end(), useTaskId) != idTasksMultiple.end());
            assert(0 < nbTasksInUsed);
            assert(nbTasksReleased < nbTasksInUsed);
            nbTasksReleased += 1;
            assert(nbTasksReleased <= int(idTasksMultiple.size()));
            if(accessMode == SpDataAccessMode::COMMUTATIVE_WRITE){
                assert(nbTasksReleased == nbTasksInUsed);
                // Return true if there still are any unfulfilled commutative write access requests
                // on the data handle. So basically, by returning true in this case we notify the caller
                // that the data handle is now available for another task to request its
                // commutative write access onto. If all commutative write access requests on the data handle have
                // been fulfilled we return false, basically saying this dependency slot
                // is not available for any further memory accesses and we should move onto the next dependency slot.
                return (nbTasksReleased != int(idTasksMultiple.size()));
            }
            else{
                // Tasks that want to read can read however they please,
                // they don't have to wait for a previous task reading from the data handle
                // to give them permission to access to the data handle through the dependency slot.
                // So basically we are saying read requests are already "fulfilled" by default (note that this does not
                // necessarily mean that the read request has already been released from the dependency slot).
                return false;
            }
        }
    }

    //! Return true if all tasks have used the dependence (and finished to use)
    //! Must be called after release
    bool isOver() const{
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            assert(nbTasksReleased == 1);
            assert(nbTasksInUsed == 1);
            return true;
        }
        else{
            assert(0 < nbTasksInUsed);
            assert(0 < nbTasksReleased);
            assert(nbTasksReleased <= nbTasksInUsed);
            return (nbTasksReleased == static_cast<long int>(idTasksMultiple.size()));
        }
    }

    //! If not over, return true if can be used by another task now
    //! Must be called after release
    bool isAvailable() const {
        assert(isOver() == false);
        assert(nbTasksReleased <= nbTasksInUsed);
        if(accessMode == SpDataAccessMode::WRITE || accessMode == SpDataAccessMode::POTENTIAL_WRITE){
            return nbTasksInUsed == 0;
        }
        else if(accessMode == SpDataAccessMode::COMMUTATIVE_WRITE){
            return nbTasksInUsed == nbTasksReleased && (nbTasksReleased < static_cast<long int>(idTasksMultiple.size()));
        }
        else{
            return (nbTasksInUsed < static_cast<long int>(idTasksMultiple.size()));
        }
    }
};

#endif
