#ifndef SPCUDAMEMMANAGER_HPP
#define SPCUDAMEMMANAGER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPECX_COMPILE_WITH_CUDA
#error SPECX_COMPILE_WITH_CUDA must be defined
#endif

#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include <cstring>

#include <cuda_runtime.h>
#include <cuda.h>

#include <Utils/small_vector.hpp>
#include <Data/SpAbstractDeviceMemManager.hpp>
#include <Utils/SpConsumerThread.hpp>
#include "SpCudaUtils.hpp"

class SpCudaManager {
    struct DataObj{
        void* ptr;
        std::size_t size;
    };

    struct HandleDescr{
        small_vector<DataObj> groupOfBlocks;
        std::list<void*>::iterator lruIterator;
        int useCount = 0;
    };

    static std::mutex CudaMutex;

public:

    static void Lock(){// Do finer lock TODO
        CudaMutex.lock();
    }

    static void Unlock(){
        CudaMutex.unlock();
    }

    class SpCudaMemManager : public SpAbstractDeviceMemManager {
        const int id;
        std::unordered_map<void*, HandleDescr> handles;
        std::unordered_map<const void*, DataObj> allBlocks;

        std::list<void*> lru;

        cudaStream_t extraStream;
        std::unique_ptr<SpConsumerThread> deferCopier;
        size_t remainingMemory;

    public:
        explicit SpCudaMemManager(const int inId)
            : id(inId), deferCopier(new SpConsumerThread), remainingMemory(0){

            deferCopier->submitJobAndWait([this]{
                SpCudaUtils::UseDevice(id);
                CUDA_ASSERT(cudaStreamCreate(&extraStream));
                remainingMemory = size_t(double(SpCudaUtils::GetFreeMemOnDevice())*0.8);
            });
        }

        ~SpCudaMemManager(){
            if(deferCopier){
                assert(handles.size() == 0);
                assert(allBlocks.size() == 0);
                // In in release and data remain, just delete them
                deferCopier->submitJobAndWait([this]{
                    for(auto& handle : handles){
                        assert(handle.second.useCount == 0);
                        for(auto& block : handle.second.groupOfBlocks){
                            CUDA_ASSERT(cudaFreeAsync(block.ptr, extraStream));
                            remainingMemory += block.size;
                        }
                    }
                    CUDA_ASSERT(cudaStreamDestroy(extraStream));
                });
            }
        }

        SpCudaMemManager(const SpCudaMemManager&) = delete;
        SpCudaMemManager(SpCudaMemManager&&) = default;

        SpCudaMemManager& operator=(const SpCudaMemManager&) = delete;
        SpCudaMemManager& operator=(SpCudaMemManager&&) = delete;

        void incrDeviceDataUseCount(void* key) override {
            assert(handles.find(key) != handles.end());
            handles[key].useCount += 1;
            if(handles[key].lruIterator != lru.begin()){
                lru.erase(handles[key].lruIterator);
                lru.push_front(key);
                handles[key].lruIterator = lru.begin();
            }
        }

        void decrDeviceDataUseCount(void* key) override {
            assert(handles.find(key) != handles.end());
            handles[key].useCount -= 1;
        }

        bool hasBeenRemoved(void* key){
            return (handles.find(key) == handles.end());
        }

        bool hasEnoughSpace(std::size_t inByteSize) override{
            // SpCudaUtils::GetFreeMemOnDevice() cannot be used
            // because it is not up to date
            return inByteSize <= remainingMemory;
        }

        std::list<void*> candidatesToBeRemoved(const std::size_t inByteSize) override{
            std::list<void*> candidates;
            std::size_t toBeReleased = 0;
            const auto iterend = lru.rend();
            for(auto iter = lru.rbegin() ; iter != iterend && toBeReleased < inByteSize ; ){
                void* handleToRemove = (*iter);
                if(handles[handleToRemove].useCount == 0){
                    for(auto block : handles[handleToRemove].groupOfBlocks){
                        assert(block.ptr);
                        toBeReleased += block.size;
                    }
                    candidates.push_back(handleToRemove);
                }
                ++iter;
            }
            return candidates;
        }

        void* allocateWithKey(void* key, std::size_t inByteSize,
                              std::size_t /*alignment*/) override{
            assert(hasEnoughSpace(inByteSize));
            DataObj data;
            data.size = inByteSize;
            assert(data.size <= remainingMemory);
            remainingMemory -= inByteSize;

            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMallocAsync(&data.ptr, inByteSize, SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMallocAsync(&data.ptr, inByteSize, extraStream));
                });
            }

            allBlocks[data.ptr] = data;
            if(handles.find(key) == handles.end()){
                lru.push_front(key);
                handles[key].lruIterator = lru.begin();
            }
            handles[key].groupOfBlocks.push_back(data);
            return data.ptr;
        }

        std::size_t freeGroup(void* key) override{
            assert(handles.find(key) != handles.end());
            assert(handles[key].useCount == 0);
            lru.erase(handles[key].lruIterator);

            std::size_t released = 0;
            for(auto& data : handles[key].groupOfBlocks){
                released += data.size;

                if(SpCudaUtils::CurrentWorkerIsCuda()){
                    CUDA_ASSERT(cudaFreeAsync(data.ptr, SpCudaUtils::GetCurrentStream()));
                    CUDA_ASSERT(cudaStreamSynchronize(SpCudaUtils::GetCurrentStream()));
                }
                else{
                    deferCopier->submitJobAndWait([&,this]{
                        CUDA_ASSERT(cudaFreeAsync(data.ptr, extraStream));
                        CUDA_ASSERT(cudaStreamSynchronize(extraStream));
                    });
                }
                assert(allBlocks.find(data.ptr) != allBlocks.end());
                allBlocks.erase(data.ptr);
            }
            remainingMemory += released;

            handles.erase(key);
            return released;
        }

        void memset(void* inPtrDev, const int val, const std::size_t inByteSize) override{
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMemsetAsync(inPtrDev, val, inByteSize, SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMemsetAsync(inPtrDev, val, inByteSize, extraStream));
                });
            }
        }

        void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize)  override {
            // The following assert cannot be use as it will fire if we work on a sub-block
            // maybe we could iterate to find the block(?)
            //assert(allBlocks.find(inPtrDev) != allBlocks.end() && inByteSize <= allBlocks[inPtrDev].size);
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMemcpyAsync(inPtrDev, inPtrHost, inByteSize, cudaMemcpyHostToDevice,
                                        SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMemcpyAsync(inPtrDev, inPtrHost, inByteSize, cudaMemcpyHostToDevice,
                                            extraStream));
                });
            }
        }

        void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize)  override{
            // The following assert it not valid as inPtrDev might be a subblock
            // assert(allBlocks.find(inPtrDev) != allBlocks.end()
            //        && allBlocks[inPtrDev].size <= inByteSize);
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMemcpyAsync(inPtrHost, inPtrDev, inByteSize, cudaMemcpyDeviceToHost,
                                        SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMemcpyAsync(inPtrHost, inPtrDev, inByteSize, cudaMemcpyDeviceToHost,
                                            extraStream));
                });
            }
        }

        void copyDeviceToDevice(void* inPtrDevDest, const void* inPtrDevSrc, const int srcId,
                                const std::size_t inByteSize)  override{
            assert(allBlocks.find(inPtrDevDest) != allBlocks.end()
                    && allBlocks[inPtrDevDest].size <= inByteSize);
            // This is on the other CUDA
            // assert(allBlocks.find(inPtrDevSrc) != allBlocks.end()
            //        && allBlocks[inPtrDevSrc].size <= inByteSize);
            assert(isConnectedTo(srcId));
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMemcpyPeerAsync(inPtrDevDest, id, inPtrDevSrc, srcId, inByteSize,
                                            SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMemcpyPeerAsync(inPtrDevDest, id, inPtrDevSrc, srcId, inByteSize,
                                                extraStream));
                });
            }
        }

        bool isConnectedTo(const int otherId){
            return SpCudaUtils::DevicesAreConnected(id, otherId);
        }

        void syncExtraStream(){
            deferCopier->submitJobAndWait([this]{
                SpCudaUtils::SynchronizeStream(extraStream);
            });
        }
    };

    static std::vector<SpCudaMemManager> BuildManagers(){
        std::vector<SpCudaMemManager> managers;
        const int nbCudas = SpCudaUtils::GetNbDevices();
        for(int idxCuda = 0 ; idxCuda < nbCudas ; ++idxCuda){
            managers.push_back(SpCudaMemManager(idxCuda));
        }
        return managers;
    }

    static SpCudaMemManager& GetManagerForDevice(const int deviceId){
        assert(deviceId < int(Managers.size()));
        return Managers[deviceId];
    }

    static std::vector<SpCudaMemManager> Managers;
};

#endif
