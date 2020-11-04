///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPBUFFERDATAVIEW_HPP
#define SPBUFFERDATAVIEW_HPP

#include <atomic>

#include "SpAbstractBufferManager.hpp"

template <class TargetType>
class SpDataBufferCore{
    SpAbstractBufferManager<TargetType>* bufferManager;

    std::atomic<bool> underUsage;
    std::atomic<int> dataUseLimitCounter;
    std::atomic<TargetType*> dataPtr;
    std::atomic<int> dataUseCounter;

    std::atomic<int> nbOfPossibleDeleter;

    bool freeDataIfPossible(){
        if(underUsage == false && dataUseLimitCounter == dataUseCounter){
            TargetType* todeleteDataPtr = dataPtr;
            TargetType* nullDataptr = nullptr;
            if(todeleteDataPtr != nullptr && dataPtr.compare_exchange_strong(todeleteDataPtr, nullDataptr) == true){
                assert(nbOfPossibleDeleter >= 1);
                while(nbOfPossibleDeleter != 1);
                if(bufferManager){
                    bufferManager->releaseABuffer(todeleteDataPtr);
                }
                else{
                    delete todeleteDataPtr;
                }
                delete this;
                return true;
            }
            else{
                assert(todeleteDataPtr == nullptr);
            }
        }
        return false;
    }

    ~SpDataBufferCore(){
    }

public:
    explicit SpDataBufferCore(SpAbstractBufferManager<TargetType>* inBufferManager = nullptr)
        : bufferManager(inBufferManager), underUsage(true), dataUseLimitCounter(0), dataPtr(nullptr), dataUseCounter(0), nbOfPossibleDeleter(0){
    }

    SpDataBufferCore(const SpDataBufferCore&) = delete;
    SpDataBufferCore(SpDataBufferCore&&) = delete;
    SpDataBufferCore& operator=(const SpDataBufferCore&) = delete;
    SpDataBufferCore& operator=(SpDataBufferCore&&) = delete;

    void useData(){
        // Do nothing
        assert(dataUseCounter < dataUseLimitCounter);
    }

    TargetType* getData(){
        if(dataPtr == nullptr){
            TargetType* newDataPtr = (bufferManager ? bufferManager->getABuffer() : new TargetType());
            TargetType* nullDataptr = nullptr;
            if(dataPtr.compare_exchange_strong(nullDataptr, newDataPtr) == false){
                assert(nullDataptr != nullptr);
                if(bufferManager){
                    bufferManager->releaseABuffer(newDataPtr);
                }
                else{
                    delete newDataPtr;
                }
            }
        }
        return dataPtr;
    }

    void releaseData(){
        nbOfPossibleDeleter += 1;
        assert(dataUseCounter < dataUseLimitCounter);
        dataUseCounter += 1;
        if(freeDataIfPossible() == false){
            nbOfPossibleDeleter -= 1;
        }
    }

    void addOneUse(){
        assert(underUsage);
        dataUseLimitCounter += 1;
    }

    void dataUseLimitIsFixed(){
        nbOfPossibleDeleter += 1;
        assert(underUsage);
        underUsage = false;
        if(freeDataIfPossible() == false){
            nbOfPossibleDeleter -= 1;
        }
    }
};


template <class TargetType>
class SpDataBuffer{
    SpDataBufferCore<TargetType>* dataPtr;

public:
    SpDataBuffer(SpDataBufferCore<TargetType>& inData)
        : dataPtr(&inData){
        dataPtr->useData();
    }

    SpDataBuffer(const SpDataBufferCore<TargetType>& inData)
        : SpDataBuffer(*const_cast<SpDataBufferCore<TargetType>*>(&inData)){
    }

    SpDataBuffer(const SpDataBuffer&) = delete;
    SpDataBuffer& operator=(const SpDataBuffer&) = delete;

    ~SpDataBuffer(){
        dataPtr->releaseData();
    }

    operator TargetType&(){
        return *dataPtr->getData();
    }

    operator const TargetType&() const{
        return *dataPtr->getData();
    }

    TargetType* operator->(){
        return dataPtr->getData();
    }

    const TargetType* operator->() const{
        return dataPtr->getData();
    }

    TargetType& operator*(){
        return *dataPtr->getData();
    }

    const TargetType& operator*() const{
        return *dataPtr->getData();
    }
};

template <class TargetType>
class SpBufferDataView{
    SpDataBufferCore<TargetType>* dataPtr;

public:
    explicit SpBufferDataView(SpAbstractBufferManager<TargetType>* inBufferManager = nullptr){
        dataPtr = new SpDataBufferCore<TargetType>(inBufferManager);
    }

    SpBufferDataView(SpBufferDataView&&) = default;

    SpBufferDataView(const SpBufferDataView&) = delete;
    SpBufferDataView& operator=(const SpBufferDataView&) = delete;
    SpBufferDataView& operator=(SpBufferDataView&&) = delete;

    ~SpBufferDataView(){
        dataPtr->dataUseLimitIsFixed();
    }

    // Must be called only when passed to a task
    SpDataBufferCore<TargetType>& getDataDep(){
        dataPtr->addOneUse();
        return *dataPtr;
    }
};

#endif
