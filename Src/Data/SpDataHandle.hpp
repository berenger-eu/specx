///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDATAHANDLE_HPP
#define SPDATAHANDLE_HPP

#include <vector>
#include <cassert>
#include <mutex>
#include <atomic>

#include "Config/SpConfig.hpp"
#include "SpDependence.hpp"
#include "Utils/SpUtils.hpp"
#include "Utils/small_vector.hpp"

#ifdef SPECX_COMPILE_WITH_CUDA
#include "Data/SpDeviceData.hpp"
#include "Data/SpDataDuplicator.hpp"
#include "Cuda/SpCudaMemManager.hpp"
#endif // SPECX_COMPILE_WITH_CUDA

#ifdef SPECX_COMPILE_WITH_HIP
#include "Data/SpDeviceData.hpp"
#include "Data/SpDataDuplicator.hpp"
#include "Hip/SpHipMemManager.hpp"
#endif // SPECX_COMPILE_WITH_HIP

//! This is a register data to apply the
//! dependences on it.
class SpDataHandle {
private:
    //! Generic pointer to the data
    void* ptrToData;
#ifdef SPECX_COMPILE_WITH_CUDA
    //! Copy of the CPU object on CUDAs
    std::array<SpDeviceData, SpConfig::SpMaxNbCudas> copies;
    //! Copy builder from/to CPU/CUDA
    std::unique_ptr<SpAbstractDeviceDataCopier<SpCudaManager::SpCudaMemManager>> deviceDataOp;
    //! Tell if the CPU version is OK
    bool cpuDataOk;
#endif // SPECX_COMPILE_WITH_CUDA
#ifdef SPECX_COMPILE_WITH_HIP
    //! Copy of the CPU object on CUDAs
    std::array<SpDeviceData, SpConfig::SpMaxNbHips> copies;
    //! Copy builder from/to CPU/CUDA
    std::unique_ptr<SpAbstractDeviceDataCopier<SpHipManager::SpHipMemManager>> deviceDataOp;
    //! Tell if the CPU version is OK
    bool cpuDataOk;
#endif // SPECX_COMPILE_WITH_CUDA
    //! Lock the data
    std::mutex handleLock;

    //! All the dependences on the current data
    small_vector<SpDependence, 4> dependencesOnData;

    //! To ensure safe access to the dependencesOnData vector
    mutable std::mutex mutexDependences;

    //! Current execution state in the dependences list
    long int currentDependenceCursor;

public:
    template <class DataType>
    explicit SpDataHandle(DataType* inPtrToData)
        : ptrToData(inPtrToData),
      #ifdef SPECX_COMPILE_WITH_CUDA
          copies(),
          deviceDataOp(new SpDeviceDataCopier<DataType, SpCudaManager::SpCudaMemManager>()),
          cpuDataOk(true),
      #endif
      #ifdef SPECX_COMPILE_WITH_HIP
          copies(),
          deviceDataOp(new SpDeviceDataCopier<DataType, SpHipManager::SpHipMemManager>()),
          cpuDataOk(true),
      #endif
        dependencesOnData(), mutexDependences(), currentDependenceCursor(0){

        if(SpDebug::Controller.isEnable()){
            const std::string datatypeName(typeid(DataType).name());
            SpDebugPrint() << "[SpDataHandle] Create handle for data " << inPtrToData << " of type " << datatypeName;
        }
        #if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
        assert(deviceDataOp != nullptr && "deviceDataOp is not initialized");
        #endif
    }
    
    //! Cannot be copied or moved
    SpDataHandle(const SpDataHandle&) = delete;
    SpDataHandle(SpDataHandle&&) = delete;
    SpDataHandle& operator=(const SpDataHandle&) = delete;
    SpDataHandle& operator=(SpDataHandle&&) = delete;

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
    virtual ~SpDataHandle(){
        #if defined(SPECX_COMPILE_WITH_CUDA)
        syncCpuDataIfNeeded(SpCudaManager::Managers);
        setCpuOnlyValid(SpCudaManager::Managers);
        #endif
        #if defined(SPECX_COMPILE_WITH_HIP)
        syncCpuDataIfNeeded(SpHipManager::Managers);
        setCpuOnlyValid(SpHipManager::Managers);
        #endif
    }

    template <class Allocators>
    void setCpuOnlyValid(Allocators& memManagers) {
        assert(cpuDataOk == true);
        for(int idxGpu = 0 ; idxGpu < int(copies.size()) ; ++idxGpu){
            if(copies[idxGpu].ptr){
                deviceDataOp->freeGroup(memManagers[idxGpu], this, ptrToData);
                copies[idxGpu] = SpDeviceData();
            }
        }
    }

    template <class Allocators>
    void setGpuOnlyValid(Allocators& memManagers, const int gpuId) {
        assert(gpuId < memManagers.size() && gpuId < int(copies.size()));
        assert(copies[gpuId].ptr);
        cpuDataOk = false;
        for(int idxGpu = 0 ; idxGpu < int(copies.size()) ; ++idxGpu){
            if(idxGpu != gpuId && copies[idxGpu].ptr){
                deviceDataOp->freeGroup(memManagers[idxGpu], this, ptrToData);
                copies[idxGpu] = SpDeviceData();
            }
        }
    }

    template <class Allocators>
    int syncCpuDataIfNeeded(Allocators& memManagers){
        if(cpuDataOk == false){
            auto idxGpuSrcIter = std::find_if(copies.begin(), copies.end(), [](auto iter) -> bool {
                return iter.ptr != nullptr;
            });
            assert(idxGpuSrcIter != copies.end());
            const int idxGpu = int(std::distance(copies.begin(), idxGpuSrcIter));
            deviceDataOp->copyDeviceToHost(memManagers[idxGpu], this, ptrToData, copies[idxGpu]);
            cpuDataOk = true;
            return idxGpu;
        }
        return -1;
    }

    template <class Allocators>
    void removeFromGpu(Allocators& memManagers, const int gpuId){
        assert(gpuId < memManagers.size() && gpuId < int(copies.size()));
        assert(copies[gpuId].ptr);
        syncCpuDataIfNeeded(memManagers);
        deviceDataOp->freeGroup(memManagers[gpuId], this, ptrToData);
        copies[gpuId] = SpDeviceData();
    }

    template <class Allocators>
    SpDeviceData& getDeviceData(Allocators& memManagers, const int gpuId) {
        assert(gpuId < memManagers.size() && gpuId < int(copies.size()));
#if defined(SPECX_COMPILE_WITH_CUDA)
        assert(gpuId < SpConfig::SpMaxNbCudas);
#elif defined(SPECX_COMPILE_WITH_HIP)
        assert(gpuId < SpConfig::SpMaxNbHips);
#endif
        if(copies[gpuId].ptr == nullptr || memManagers[gpuId].hasBeenRemoved(this)){
            copies[gpuId].ptr = nullptr;
            auto idxGpuSrcIter = std::find_if(copies.begin(), copies.end(), [](auto iter) -> bool {
                return iter.ptr != nullptr;
            });
            if(!deviceDataOp->hasEnoughSpace(memManagers[gpuId], this, ptrToData)){
                auto candidates = deviceDataOp->candidatesToBeRemoved(memManagers[gpuId], this, ptrToData);
                for(auto toRemove : candidates){
                    assert(toRemove != this);
                    reinterpret_cast<SpDataHandle*>(toRemove)->lock();
                    reinterpret_cast<SpDataHandle*>(toRemove)->removeFromGpu(memManagers, gpuId);
                    reinterpret_cast<SpDataHandle*>(toRemove)->unlock();
                }
            }
            copies[gpuId] = deviceDataOp->allocate(memManagers[gpuId], this, ptrToData);
            if(idxGpuSrcIter != copies.end()){
                const int otherGpu = int(std::distance(copies.begin(), idxGpuSrcIter));
                if(memManagers[gpuId].isConnectedTo(otherGpu)){
                    copies[gpuId].viewPtr = deviceDataOp->copyFromDeviceToDevice(memManagers[gpuId], this, ptrToData, copies[gpuId], copies[otherGpu], otherGpu);
                }
                else{
                    syncCpuDataIfNeeded(memManagers);

                   copies[gpuId].viewPtr = deviceDataOp->copyHostToDevice(memManagers[gpuId], this, copies[gpuId], ptrToData);
                }
            }
            else{
                assert(cpuDataOk);
                copies[gpuId].viewPtr = deviceDataOp->copyHostToDevice(memManagers[gpuId], this, copies[gpuId], ptrToData);
            }
        }
        return copies[gpuId];
    }
#endif // SPECX_COMPILE_WITH_CUDA || SPECX_COMPILE_WITH_HIP

	void lock() {
		handleLock.lock();
	}
    
    void unlock() {
		handleLock.unlock();
	}
    
    void* getRawPtr() {
        return ptrToData;
    }

    //! Convert to pointer to Datatype
    template <class Datatype>
    std::remove_reference_t<Datatype>* castPtr(){
        return reinterpret_cast<std::remove_reference_t<Datatype>*>(ptrToData);
    }

    //! Add a new dependence to the data
    //! it returns the dependence position
    long int addDependence(SpAbstractTask* inTask, const SpDataAccessMode inAccessMode){
        // protect dependencesOnData
        std::unique_lock<std::mutex> lock(mutexDependences);
        // If no dependence exist, or they have been all consumed already, or the new dependence
        // mode is difference from the last one
        if(dependencesOnData.size() == 0 || currentDependenceCursor == static_cast<long int>(dependencesOnData.size())
                || inAccessMode != dependencesOnData.back().getMode()){
            // Add a new one
            dependencesOnData.emplace_back(inTask, inAccessMode);
        }
        // From here we now that the new dependence mode is the same as the last one
        else if(inAccessMode == SpDataAccessMode::WRITE){
            assert(dependencesOnData.back().getMode() == SpDataAccessMode::WRITE);
            // Write cannot not be done concurently, so create a new dependence
            dependencesOnData.emplace_back(inTask, inAccessMode);
        }
        else if(inAccessMode == SpDataAccessMode::POTENTIAL_WRITE){
            assert(dependencesOnData.back().getMode() == SpDataAccessMode::POTENTIAL_WRITE);
            // Write cannot not be done concurently, so create a new dependence
            dependencesOnData.emplace_back(inTask, inAccessMode);
        }
        else if(inAccessMode == SpDataAccessMode::PARALLEL_WRITE){
            assert(dependencesOnData.back().getMode() == SpDataAccessMode::PARALLEL_WRITE);
            // append to the last dependence, because can be done concurrently
            dependencesOnData.back().addTaskForMultiple(inTask);
        }
        else if(inAccessMode == SpDataAccessMode::COMMUTATIVE_WRITE){
            assert(dependencesOnData.back().getMode() == SpDataAccessMode::COMMUTATIVE_WRITE);
            // append to the last dependence, because can be done concurrently
            dependencesOnData.back().addTaskForMultiple(inTask);
        }
        else /*if(inAccessMode == SpDataAccessMode::READ && dependencesOnData.back().getMode() == SpDataAccessMode::READ)*/{
            assert(inAccessMode == SpDataAccessMode::READ);
            assert(dependencesOnData.back().getMode() == SpDataAccessMode::READ);
            // append to the last dependence, because can be done concurrently
            dependencesOnData.back().addTaskForMultiple(inTask);
        }
        // Return the corresponding dependence idx
        return static_cast<int>(dependencesOnData.size()-1);
    }
    
    //! Return the data mode access for dependence at position inDependenceIdx
    SpDataAccessMode getModeByTask(const long int inDependenceIdx) const {
        std::unique_lock<std::mutex> lock(mutexDependences);
        assert(inDependenceIdx < static_cast<long int>(dependencesOnData.size()));
        return dependencesOnData[inDependenceIdx].getMode();
    }

    //! To know if a dependence is ready for a given task
    bool canBeUsedByTask(const SpAbstractTask* inTask, const long int inDependenceIdx) const {
        std::unique_lock<std::mutex> lock(mutexDependences);
        assert(inDependenceIdx < static_cast<long int>(dependencesOnData.size()));
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpDataHandle] " << this << " canBeUsedByTask inDependenceIdx " << inDependenceIdx << " currentDependenceCursor " << currentDependenceCursor
                       << " mode " << SpModeToStr(dependencesOnData[inDependenceIdx].getMode()) << " address " << ptrToData;
        }
        assert(currentDependenceCursor <= inDependenceIdx);
        // Return true if current execution cursor is set to the dependence and that dependence return true to canBeUsedByTask
        if(inDependenceIdx == currentDependenceCursor && dependencesOnData[inDependenceIdx].canBeUsedByTask(inTask)){
            return true;
        }
        return false;
    }

    //! Mark the dependence as used by the given task
    //! canBeUsedByTask must be true for the same parameter
    void setUsedByTask(SpAbstractTask* inTask, const long int inDependenceIdx){
        assert(canBeUsedByTask(inTask, inDependenceIdx));
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpDataHandle] " << this << " setUsedByTask inDependenceIdx " << inDependenceIdx << " currentDependenceCursor " << currentDependenceCursor
                       << " mode " << SpModeToStr(dependencesOnData[inDependenceIdx].getMode()) << " address " << ptrToData;
        }
        std::unique_lock<std::mutex> lock(mutexDependences);
        dependencesOnData[inDependenceIdx].setUsedByTask(inTask);
    }

    //! Release the dependence, the dependence must have been set to used by
    //! the input task.
    //! The method returns true if there still are any unfulfilled memory access
    //! requests on the data handle.
    bool releaseByTask(SpAbstractTask* inTask, const long int inDependenceIdx){
        std::unique_lock<std::mutex> lock(mutexDependences);
        assert(inDependenceIdx == currentDependenceCursor);
        if(SpDebug::Controller.isEnable()){
            SpDebugPrint() << "[SpDataHandle] " << this << " releaseByTask " << inTask << " inDependenceIdx " << inDependenceIdx << " address " << ptrToData;
        }
        // Release memory access request on dependency slot. As a return value we get a boolean flag
        // telling us if there are still any unfulfilled memory access requests registered on the dependency slot.
        const bool thereStillAreUnfulfilledMemoryAccessRequestsOnTheDependencySlot = dependencesOnData[inDependenceIdx].releaseByTask(inTask);
        
        // Have all memory accesses registered on the dependency slot been performed on the data ?
        if(dependencesOnData[inDependenceIdx].isOver()){
            assert(thereStillAreUnfulfilledMemoryAccessRequestsOnTheDependencySlot == false);
            
            // Move on to the next dependency slot
            currentDependenceCursor += 1;
            
            // Mask as available
            if(SpDebug::Controller.isEnable()){
                SpDebugPrint() << "[SpDataHandle] releaseByTask isOver dependencesOnData true currentDependenceCursor " << currentDependenceCursor << " dependencesOnData.size() " << dependencesOnData.size();
            }
            // Return true if we still have not fulfilled all memory access requests on the data handle  
            return (currentDependenceCursor != static_cast<long int>(dependencesOnData.size()));
            
        } else { // There still are unfulfilled or unreleased memory access requests registered on the dependency slot
            SpDebugPrint() << "[SpDataHandle] releaseByTask isAvailable dependencesOnData true currentDependenceCursor " 
                           << currentDependenceCursor << " dependencesOnData.size() " << dependencesOnData.size();
            return thereStillAreUnfulfilledMemoryAccessRequestsOnTheDependencySlot;
        }
        
        // All memory access requests on the data handle have been released.
        return false;
    }

    //! Get the potential ready tasks on the current cursor
    void fillCurrentTaskList(small_vector_base<SpAbstractTask*>* potentialReady) const {
        std::unique_lock<std::mutex> lock(mutexDependences);
        if(currentDependenceCursor != static_cast<long int>(dependencesOnData.size())) {
            dependencesOnData[currentDependenceCursor].fillWithListOfPotentiallyReadyTasks(potentialReady);
        }
    }

    //! Get the list of tasks that depend on dependence at idx afterIdx
    void getDependences(small_vector_base<SpAbstractTask*>* dependences, const long int afterIdx) const {
        std::unique_lock<std::mutex> lock(mutexDependences);
        if(afterIdx != static_cast<long int>(dependencesOnData.size()-1)){
            if(dependencesOnData[afterIdx].getMode() != SpDataAccessMode::WRITE
                    && dependencesOnData[afterIdx].getMode() != SpDataAccessMode::POTENTIAL_WRITE){
                long int skipConcatDeps = afterIdx;
                while(skipConcatDeps != static_cast<long int>(dependencesOnData.size()-1)
                      && dependencesOnData[skipConcatDeps].getMode() == dependencesOnData[skipConcatDeps+1].getMode()){
                    skipConcatDeps += 1;
                }
                if(skipConcatDeps != static_cast<long int>(dependencesOnData.size()-1)){
                    dependencesOnData[skipConcatDeps+1].fillWithTaskList(dependences);
                }
            }
            else{
                long int skipConcatDeps = afterIdx+1;
                bool isCommutativeAccess = dependencesOnData[skipConcatDeps].getMode() != SpDataAccessMode::WRITE
                                            && dependencesOnData[skipConcatDeps].getMode() != SpDataAccessMode::POTENTIAL_WRITE;
                do {
                    dependencesOnData[skipConcatDeps].fillWithTaskList(dependences);
                    skipConcatDeps += 1;
                }while(isCommutativeAccess && skipConcatDeps != static_cast<long int>(dependencesOnData.size())
                      && dependencesOnData[skipConcatDeps-1].getMode() == dependencesOnData[skipConcatDeps].getMode());
            }
        }
    }

    //! Get the list of tasks that the given task depend on
    void getPredecessors(small_vector_base<SpAbstractTask*>* dependences, const long int beforeIdx) const {
        std::unique_lock<std::mutex> lock(mutexDependences);
        if(beforeIdx != 0){
            if(dependencesOnData[beforeIdx].getMode() != SpDataAccessMode::WRITE
                    && dependencesOnData[beforeIdx].getMode() != SpDataAccessMode::POTENTIAL_WRITE){
                long int skipConcatDeps = beforeIdx-1;
                while(skipConcatDeps != -1
                      && dependencesOnData[skipConcatDeps].getMode() == dependencesOnData[skipConcatDeps+1].getMode()){
                    skipConcatDeps -= 1;
                }
                if(skipConcatDeps != -1){
                    dependencesOnData[skipConcatDeps].fillWithTaskList(dependences);
                }
            }
            else{
                dependencesOnData[beforeIdx-1].fillWithTaskList(dependences);
            }
        }
    }

};

// For variadic expansion
template <class NewType>
auto SpDataHandleToObject(SpDataHandle* inHandle) -> typename std::remove_reference<NewType>::type::HandleType {
    return *inHandle->template castPtr<typename std::remove_reference_t<NewType>::HandleType>();
}

#endif
