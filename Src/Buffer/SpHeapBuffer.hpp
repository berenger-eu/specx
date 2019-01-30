///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPHEAPBUFFER_HPP
#define SPHEAPBUFFER_HPP

#include <atomic>
#include <mutex>
#include <limits>

#include "SpAbstractBufferManager.hpp"
#include "SpBufferDataView.hpp"

template <class TargetType>
class SpHeapBuffer : public SpAbstractBufferManager<TargetType> {
    const long int softLimiteOfNbBuffers;
    std::vector<TargetType*> availableBuffers;
    std::mutex availableBuffersMutex;
    std::atomic<long int> nbBuffersUnderUse;

public:
    explicit SpHeapBuffer(const long int inSoftLimuiteOfNbBuffers = std::numeric_limits<long int>::max())
        : softLimiteOfNbBuffers(inSoftLimuiteOfNbBuffers), nbBuffersUnderUse(0){
    }

    SpHeapBuffer(const SpHeapBuffer&) = delete;
    SpHeapBuffer(SpHeapBuffer&& other) = delete;
    SpHeapBuffer& operator=(const SpHeapBuffer&) = delete;
    SpHeapBuffer& operator=(SpHeapBuffer&&) = delete;

    ~SpHeapBuffer(){
        assert(nbBuffersUnderUse == 0);
        while(availableBuffers.size()){
            delete availableBuffers.back();
            availableBuffers.pop_back();
        }
    }

    SpBufferDataView<TargetType> getNewBuffer(){
        return SpBufferDataView<TargetType>(this);
    }

    TargetType* getABuffer() final{
        std::unique_lock<std::mutex> lockAll(availableBuffersMutex);
        nbBuffersUnderUse += 1;
        if(availableBuffers.size()){
            TargetType* bufferToUse = availableBuffers.back();
            availableBuffers.pop_back();
            return bufferToUse;
        }
        else{
            return new TargetType();
        }
    }

    void releaseABuffer(TargetType* inBuffer) final{
        std::unique_lock<std::mutex> lockAll(availableBuffersMutex);
        nbBuffersUnderUse -= 1;
        if(softLimiteOfNbBuffers <= availableBuffers.size()){
            delete inBuffer;
        }
        else{
            availableBuffers.push_back(inBuffer);
        }
    }
};

#endif
