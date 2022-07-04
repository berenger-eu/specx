#ifndef SPMPISERIALIZER_HPP
#define SPMPISERIALIZER_HPP

#include <cstring>
#include <cassert>

class SpAbstractMpiSerializer {
public:
    virtual ~SpAbstractMpiSerializer(){}
    virtual unsigned char* getBuffer() = 0;
    virtual int getBufferSize() = 0;
};

template <class ObjectClass>
class SpMpiSerializer : public SpAbstractMpiSerializer {
    const ObjectClass& obj;
public:

    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){}

    virtual unsigned char* getBuffer() override{
        return reinterpret_cast<unsigned char*>(&obj);
    }
    virtual int getBufferSize() override{
        return int(sizeof (ObjectClass));
    }
};


class SpAbstractMpiDeSerializer {
public:
    virtual ~SpAbstractMpiDeSerializer(){}
    virtual void deserialize(unsigned char* buffer, int bufferSize) = 0;
};

template <class ObjectClass>
class SpMpiDeSerializer : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(unsigned char* buffer, int bufferSize) override{
        assert(bufferSize == sizeof(ObjectClass));
        memcpy(&obj, buffer, bufferSize);
    }
};

#endif // SPMPISERIALIZER_HPP
