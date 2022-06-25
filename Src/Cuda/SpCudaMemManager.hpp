#ifndef SPCUDAMEMMANAGER_HPP
#define SPCUDAMEMMANAGER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPETABARU_COMPILE_WITH_CUDA
#error CUDE MUST BE ON
#endif

#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include <cstring> // TODO remove?

#include <Utils/small_vector.hpp>
#include <Data/SpAbstractDeviceAllocator.hpp>
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

    static std::mutex GpuMutex;

public:

    static void Lock(){// Do finer lock TODO
        GpuMutex.lock();
    }

    static void Unlock(){
        GpuMutex.unlock();
    }

    class SpCudaMemManager : public SpAbstractDeviceAllocator {
        const int id;
        std::unordered_map<void*, HandleDescr> handles;
        std::unordered_map<const void*, DataObj> allBlocks;
        std::size_t memSpaceUsed;

        std::list<void*> lru;

        explicit SpCudaMemManager(const int inId)
            : id(inId){
        }

    public:
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
            return inByteSize <= GetFreeMem();
        }

        std::list<void*> candidatesToBeRemoved(const std::size_t inByteSize) override{
            std::list<void*> candidates;
            std::size_t toBeReleased = 0;
            const auto iterend = lru.rend();
            for(auto iter = lru.rbegin() ; iter != iterend && toBeReleased < inByteSize ; ){
                void* handleToRemove = (*iter);
                ++iter;
                if(handles[handleToRemove].useCount == 0){
                    for(auto block : handles[handleToRemove].groupOfBlocks){
                        toBeReleased += block.size;
                    }
                    candidates.push_back(*iter);
                }
            }
            return candidates;
        }

        void* allocateWithKey(void* key, std::size_t inByteSize,
                              std::size_t alignment) override{
            assert(hasEnoughSpace(inByteSize));
            DataObj data;
            data.size = inByteSize;
            assert(data.size <= GetFreeMem());
            // TODO CUDA_ASSERT(cudaMalloc(&data.ptr, inByteSize));
            if(alignment <= alignof(std::max_align_t)) {
                data.ptr = std::malloc(data.size);
            } else {
                data.ptr = std::aligned_alloc(data.size, alignment);
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
                // TODO CUDA_ASSERT(cudaMemFree(data.ptr));
                std::free(data.ptr);
            }

            handles.erase(key);
            return released;
        }

        void memset(void* inPtrDev, const int val, const std::size_t inByteSize) override{
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
            // TODO CUDA_ASSERT(cudaMemset(inPtrDev, val, inByteSize));
            memset(inPtrDev, val, inByteSize);
        }

        void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize)  override {
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
            std::memcpy(inPtrDev, inPtrHost, inByteSize);// TODO
        }

        void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize)  override{
            assert(allBlocks.find(inPtrDev) != allBlocks.end()
                    && allBlocks[inPtrDev].size <= inByteSize);
            std::memcpy(inPtrHost, inPtrDev, inByteSize);// TODO
        }

        void copyDeviceToDevice(void* inPtrDevDest, const void* inPtrDevSrc, const std::size_t inByteSize)  override{
            assert(allBlocks.find(inPtrDevDest) != allBlocks.end()
                    && allBlocks[inPtrDevDest].size <= inByteSize);
            assert(allBlocks.find(inPtrDevSrc) != allBlocks.end()
                    && allBlocks[inPtrDevSrc].size <= inByteSize);// TODO NO!!
            std::memcpy(inPtrDevDest, inPtrDevSrc, inByteSize);// TODO
        }

        friend SpCudaManager;
    };


    struct WorkerData {
        int gpuId;
        cudaStream_t stream;

        void init(int deviceId){
            gpuId = deviceId;
        }

        void initByWorker(){
            SpCudaUtils::UseDevice(gpuId);
            cudaStreamCreate(&stream);
        }

        void destroyByWorker(){
            cudaStreamDestroy(stream);
        }
    }


    static std::vector<SpCudaMemManager> BuildManagers(){
        std::vector<SpCudaMemManager> managers;
        const int nbGpus = GetNbCudaDevices();
        for(int idxGpu = 0 ; idxGpu < nbGpus ; ++idxGpu){
            managers.emplace_back(SpCudaMemManager(idxGpu));
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
