#ifndef SPMPISERIALIZER_HPP
#define SPMPISERIALIZER_HPP

#include <cstring>
#include <cassert>

#include "SpSerializer.hpp"

////////////////////////////////////////////////////////////////
/// The serializer class
////////////////////////////////////////////////////////////////
class SpAbstractMpiSerializer {
public:
    virtual ~SpAbstractMpiSerializer(){}
    virtual const unsigned char* getBuffer() = 0;
    virtual int getBufferSize() = 0;
};

template <class ObjectClass>
class SpMpiSerializer : public SpAbstractMpiSerializer {
    const ObjectClass& obj;
    SpSerializer serializer;
public:

    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
    	prepareSerializer();    
    }

    virtual const unsigned char* getBuffer() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();
        
        return &buffer[0];
    }
    virtual int getBufferSize() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();
        
        return buffer.size();
    }
    
private:
    
    void prepareSerializer() {
    	serializer.append(obj, "sp");
    }
};


////////////////////////////////////////////////////////////////
/// The deserializer class
////////////////////////////////////////////////////////////////
class SpAbstractMpiDeSerializer {
public:
    virtual ~SpAbstractMpiDeSerializer(){}
    virtual void deserialize(const unsigned char* buffer, int bufferSize) = 0;
};

template <class ObjectClass>
class SpMpiDeSerializer : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        SpDeserializer deserializer(&buffer[0], bufferSize);
        
        obj = deserializer.restore<ObjectClass>("sp");
    }
};

#endif // SPMPISERIALIZER_HPP
