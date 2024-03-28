#ifndef SPHIPMEMMANAGER_HPP
#define SPHIPMEMMANAGER_HPP

#include "Config/SpConfig.hpp"

#ifndef SPECX_COMPILE_WITH_HIP
#error SPECX_COMPILE_WITH_HIP must be defined
#endif

#include <mutex>
#include <list>
#include <set>
#include <unordered_set>
#include <cstring>


#include <Utils/small_vector.hpp>
#include <Data/SpAbstractDeviceMemManager.hpp>
#include <Utils/SpConsumerThread.hpp>
#include "SpHipUtils.hpp"

class SpHipManager {
    struct DataObj{
        void* ptr;
        std::size_t size;
    };

    struct HandleDescr{
        small_vector<DataObj> groupOfBlocks;
        std::list<void*>::iterator lruIterator;
        int useCount = 0;
    };

    static std::mutex HipMutex;

public:

    static void Lock(){// Do finer lock TODO
        HipMutex.lock();
    }

    static void Unlock(){
        HipMutex.unlock();
    }

    class SpHipMemManager : public SpAbstractDeviceMemManager {
        const int id;
        std::unordered_map<void*, HandleDescr> handles;
        std::unordered_map<const void*, DataObj> allBlocks;
        std::size_t memSpaceUsed;

        std::list<void*> lru;

        hipStream_t extraStream;
        std::unique_ptr<SpConsumerThread> deferCopier;

    public:
        explicit SpHipMemManager(const int inId)
            : id(inId), deferCopier(new SpConsumerThread){

            deferCopier->submitJobAndWait([this]{
                SpHipUtils::UseDevice(id);
                HIP_ASSERT(hipStreamCreate(&extraStream));
            });
        }

        ~SpHipMemManager(){
            if(deferCopier){
                deferCopier->submitJobAndWait([this]{
                    HIP_ASSERT(hipStreamDestroy(extraStream));
                });
            }
        }

        SpHipMemManager(const SpHipMemManager&) = delete;
        SpHipMemManager(SpHipMemManager&&) = default;

        SpHipMemManager& operator=(const SpHipMemManager&) = delete;
        SpHipMemManager& operator=(SpHipMemManager&&) = delete;

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
            return inByteSize <= SpHipUtils::GetFreeMemOnDevice();
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
            assert(data.size <= SpHipUtils::GetFreeMemOnDevice());
#ifndef SPECX_EMUL_HIP
            if(SpHipUtils::CurrentWorkerIsHip()){
                HIP_ASSERT(hipMalloc/*Async*/(&data.ptr, inByteSize/*, SpHipUtils::GetCurrentStream()*/));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    HIP_ASSERT(hipMalloc/*Async*/(&data.ptr, inByteSize/*, extraStream*/));
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
#ifndef SPECX_EMUL_HIP
                if(SpHipUtils::CurrentWorkerIsHip()){
                    HIP_ASSERT(hipFree/*Async*/(data.ptr/*, SpHipUtils::GetCurrentStream()*/));
                }
                else{
                    deferCopier->submitJobAndWait([&,this]{
                        HIP_ASSERT(hipFree/*Async*/(data.ptr/*, extraStream*/));
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
#ifndef SPECX_EMUL_HIP
            if(SpHipUtils::CurrentWorkerIsHip()){
                HIP_ASSERT(hipMemsetAsync(inPtrDev, val, inByteSize, SpHipUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    HIP_ASSERT(hipMemsetAsync(inPtrDev, val, inByteSize, extraStream));
                });
            }
#else
            memset(inPtrDev, val, inByteSize);
#endif
        }

        void copyHostToDevice(void* inPtrDev, const void* inPtrHost, const std::size_t inByteSize)  override {
            //assert(allBlocks.find(inPtrDev) != allBlocks.end()
            //        && allBlocks[inPtrDev].size <= inByteSize);
#ifndef SPECX_EMUL_HIP
            if(SpHipUtils::CurrentWorkerIsHip()){
                HIP_ASSERT(hipMemcpyAsync(inPtrDev, inPtrHost, inByteSize, hipMemcpyHostToDevice,
                                          SpHipUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    HIP_ASSERT(hipMemcpyAsync(inPtrDev, inPtrHost, inByteSize, hipMemcpyHostToDevice,
                                              extraStream));
                });
            }
#else
            std::memcpy(inPtrDev, inPtrHost, inByteSize);
#endif
        }

        void copyDeviceToHost(void* inPtrHost, const void* inPtrDev, const std::size_t inByteSize)  override{
            //assert(allBlocks.find(inPtrDev) != allBlocks.end()
            //        && allBlocks[inPtrDev].size <= inByteSize);
#ifndef SPECX_EMUL_HIP
            if(SpHipUtils::CurrentWorkerIsHip()){
                HIP_ASSERT(hipMemcpyAsync(inPtrHost, inPtrDev, inByteSize, hipMemcpyDeviceToHost,
                                          SpHipUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    HIP_ASSERT(hipMemcpyAsync(inPtrHost, inPtrDev, inByteSize, hipMemcpyDeviceToHost,
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
            // This is on the other HIP
            // assert(allBlocks.find(inPtrDevSrc) != allBlocks.end()
            //        && allBlocks[inPtrDevSrc].size <= inByteSize);
            assert(isConnectedTo(srcId));
#ifndef SPECX_EMUL_HIP
            if(SpHipUtils::CurrentWorkerIsHip()){
                HIP_ASSERT(hipMemcpyPeerAsync(inPtrDevDest, id, inPtrDevSrc, srcId, inByteSize,
                                              SpHipUtils::GetCurrentStream()));
            }
            else{
                deferCopier->submitJobAndWait([&,this]{
                    HIP_ASSERT(hipMemcpyPeerAsync(inPtrDevDest, id, inPtrDevSrc, srcId, inByteSize,
                                                  extraStream));
                });
            }
#else
            std::memcpy(inPtrDevDest, inPtrDevSrc, inByteSize);
#endif
        }

        bool isConnectedTo(const int otherId){
            return SpHipUtils::DevicesAreConnected(id, otherId);
        }

        void syncExtraStream(){
            deferCopier->submitJobAndWait([this]{
                SpHipUtils::SynchronizeStream(extraStream);
            });
        }
    };

    static std::vector<SpHipMemManager> BuildManagers(){
        std::vector<SpHipMemManager> managers;
        const int nbHips = SpHipUtils::GetNbDevices();
        for(int idxHip = 0 ; idxHip < nbHips ; ++idxHip){
            managers.push_back(SpHipMemManager(idxHip));
        }
        return managers;
    }

    static SpHipMemManager& GetManagerForDevice(const int deviceId){
        assert(deviceId < int(Managers.size()));
        return Managers[deviceId];
    }

    static std::vector<SpHipMemManager> Managers;
};

#endif
