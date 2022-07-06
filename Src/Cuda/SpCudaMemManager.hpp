#ifndef SPCUDAMEMMANAGER_HPP
#define SPCUDAMEMMANAGER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPECX_COMPILE_WITH_CUDA
#error CUDE MUST BE ON
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
        std::size_t memSpaceUsed;

        std::list<void*> lru;

        cudaStream_t extraStream;
        std::unique_ptr<SpConsumerThread> deferCopier;

    public:
        explicit SpCudaMemManager(const int inId)
            : id(inId), deferCopier(new SpConsumerThread){

            deferCopier->submitJobAndWait([this]{
                SpCudaUtils::UseDevice(id);
                CUDA_ASSERT(cudaStreamCreate(&extraStream));
            });
        }

        ~SpCudaMemManager(){
            if(deferCopier){
                deferCopier->submitJobAndWait([this]{
                    CUDA_ASSERT(cudaStreamDestroy(extraStream));
                });
            }
        }

        SpCudaMemManager(const SpCudaMemManager&) = delete;
        SpCudaMemManager(SpCudaMemManager&&) = default;

        SpCudaMemManager& operator=(const SpCudaMemManager&) = delete;
        SpCudaMemManager& operator=(SpCudaMemManager&&) = default;

        void incrDeviceDataUseCount(void* key){
            assert(handles.find(key) != handles.end());
            handles[key].useCount += 1;
            if(handles[key].lruIterator != lru.begin()){
                lru.erase(handles[key].lruIterator);
                lru.push_front(key);
                handles[key].lruIterator = lru.begin();
            }
        }

        void decrDeviceDataUseCount(void* key){
            assert(handles.find(key) != handles.end());
            handles[key].useCount -= 1;
        }

        bool hasBeenRemoved(void* key){
            return (handles.find(key) == handles.end());
        }

        bool hasEnoughSpace(std::size_t inByteSize) override{
            return inByteSize <= SpCudaUtils::GetFreeMemOnDevice();
        }

        std::list<void*> candidatesToBeRemoved(const std::size_t inByteSize) override{
            std::list<void*> candidates;
            std::size_t toBeReleased = 0;
            const auto iterend = lru.rend();
            for(auto iter = lru.rbegin() ; iter != iterend && toBeReleased < inByteSize ; ){
                void* handleToRemove = (*iter);
                if(handles[handleToRemove].useCount == 0){
                    for(auto block : handles[handleToRemove].groupOfBlocks){
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
            assert(data.size <= SpCudaUtils::GetFreeMemOnDevice());
#ifndef SPECX_EMUL_CUDA
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMallocAsync(&data.ptr, inByteSize, SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMallocAsync(&data.ptr, inByteSize, extraStream));
                });
            }
#else
            if(alignment <= alignof(std::max_align_t)) {
                data.ptr = std::malloc(data.size);
            } else {
                data.ptr = std::aligned_alloc(data.size, alignment);
            }
#endif
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
#ifndef SPECX_EMUL_CUDA
                if(SpCudaUtils::CurrentWorkerIsCuda()){
                    CUDA_ASSERT(cudaFreeAsync(data.ptr, SpCudaUtils::GetCurrentStream()));
                }
                else{
                    deferCopier->submitJobAndWait([&,this]{
                        CUDA_ASSERT(cudaFreeAsync(data.ptr, extraStream));
                    });
                }
#else
                std::free(data.ptr);
#endif
            }

            handles.erase(key);
            return released;
        }

        void memset(void* inPtrDev, const int val, const std::size_t inByteSize) override{
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
#ifndef SPECX_EMUL_CUDA
            if(SpCudaUtils::CurrentWorkerIsCuda()){
                CUDA_ASSERT(cudaMemsetAsync(inPtrDev, val, inByteSize, SpCudaUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    CUDA_ASSERT(cudaMemsetAsync(inPtrDev, val, inByteSize, extraStream));
                });
            }
#else
            memset(inPtrDev, val, inByteSize);
#endif
        }

        void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize)  override {
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
#ifndef SPECX_EMUL_CUDA
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
#else
            std::memcpy(inPtrDev, inPtrHost, inByteSize);
#endif
        }

        void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize)  override{
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
#ifndef SPECX_EMUL_CUDA
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
#else
            std::memcpy(inPtrHost, inPtrDev, inByteSize);
#endif
        }

        void copyDeviceToDevice(void* inPtrDevDest, const void* inPtrDevSrc, const int srcId,
                                const std::size_t inByteSize)  override{
            assert(allBlocks.find(inPtrDevDest) != allBlocks.end()
                    && allBlocks[inPtrDevDest].size <= inByteSize);
            // This is on the other CUDA
            // assert(allBlocks.find(inPtrDevSrc) != allBlocks.end()
            //        && allBlocks[inPtrDevSrc].size <= inByteSize);
            assert(isConnectedTo(srcId));
#ifndef SPECX_EMUL_CUDA
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
#else
            std::memcpy(inPtrDevDest, inPtrDevSrc, inByteSize);
#endif
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
