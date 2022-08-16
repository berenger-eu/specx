#ifndef SPMPISERIALIZER_HPP
#define SPMPISERIALIZER_HPP

#include <cstring>
#include <cassert>

#include "SpSerializer.hpp"

enum class SpSerializationType{
    SP_SERIALIZER_TYPE,
    SP_RAW_TYPE,
    SP_DIRECT_ACCESS,
    SP_VEC_SERIALIZER_TYPE,
    SP_VEC_RAW_TYPE,
    SP_VEC_DIRECT_ACCESS,
    SP_UNDEFINED_TYPE
};

class SpGetSerializationTypeHelper{
public:
    template <typename, typename = std::void_t<>>
    struct has_getRawData
    : public std::false_type {};

    template <typename Class>
    struct has_getRawData<Class,
        std::void_t<decltype(std::declval<Class>().getRawData())>>
    : public std::is_same<decltype(std::declval<Class>().getRawData()), const unsigned char*>
    {};

    template <typename, typename = std::void_t<>>
    struct has_getRawDataSize
    : public std::false_type {};

    template <typename Class>
    struct has_getRawDataSize<Class,
        std::void_t<decltype(std::declval<Class>().getRawDataSize())>>
    : public std::is_same<decltype(std::declval<Class>().getRawDataSize()), std::size_t>
    {};

    template <typename, typename = std::void_t<>>
    struct has_restoreRawData
    : public std::false_type {};

    template <typename Class>
    struct has_restoreRawData<Class,
        std::void_t<decltype(std::declval<Class>().restoreRawData(std::declval<const unsigned char*>(), std::declval<std::size_t>()))>>
    : public std::is_same<decltype(std::declval<Class>().restoreRawData(std::declval<const unsigned char*>(), std::declval<std::size_t>())), void>
    {};

    template<class T> struct is_stdvector : public std::false_type {};

    template<class T, class Alloc>
    struct is_stdvector<std::vector<T, Alloc>> : public std::true_type {
        using _T = T;
    };
};

template <class ObjectClass>
constexpr auto SpGetSerializationType(){
    using ObjectClassClean = typename std::decay<ObjectClass>::type;
    if constexpr(std::is_base_of_v<SpAbstractSerializable, ObjectClassClean>){
        return SpSerializationType::SP_SERIALIZER_TYPE;
    }
    if constexpr (SpGetSerializationTypeHelper::has_getRawDataSize<ObjectClassClean>::value
                    && SpGetSerializationTypeHelper::has_getRawData<ObjectClassClean>::value
                    && SpGetSerializationTypeHelper::has_restoreRawData<ObjectClassClean>::value){
        return SpSerializationType::SP_DIRECT_ACCESS;
    }
    if constexpr(std::is_standard_layout_v<ObjectClassClean> && std::is_trivial_v<ObjectClassClean>){
        return SpSerializationType::SP_RAW_TYPE;
    }
    if constexpr(SpGetSerializationTypeHelper::is_stdvector<ObjectClassClean>::value){
        constexpr SpSerializationType typeChild = SpGetSerializationType<typename ObjectClassClean::value_type>();
        if constexpr(typeChild == SpSerializationType::SP_SERIALIZER_TYPE){
            return SpSerializationType::SP_VEC_SERIALIZER_TYPE;
        }
        else if constexpr(typeChild == SpSerializationType::SP_RAW_TYPE){
            return SpSerializationType::SP_VEC_RAW_TYPE;
        }
        else if constexpr(typeChild == SpSerializationType::SP_DIRECT_ACCESS){
            return SpSerializationType::SP_VEC_DIRECT_ACCESS;
        }
    }
    return SpSerializationType::SP_UNDEFINED_TYPE;
}

////////////////////////////////////////////////////////////////
/// The serializer class
////////////////////////////////////////////////////////////////
class SpAbstractMpiSerializer {
public:
    virtual ~SpAbstractMpiSerializer(){}
    virtual const unsigned char* getBuffer() = 0;
    virtual int getBufferSize() = 0;
};

template <SpSerializationType type, class ObjectClass>
class SpMpiSerializer;

template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_SERIALIZER_TYPE, ObjectClass> : public SpAbstractMpiSerializer {
    const ObjectClass& obj;
    SpSerializer serializer;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
        serializer.append(obj, "sp");
    }

    virtual const unsigned char* getBuffer() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();        
        return &buffer[0];
    }
    virtual int getBufferSize() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();        
        return buffer.size();
    }
};


template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_RAW_TYPE, ObjectClass> : public SpAbstractMpiSerializer {
    const ObjectClass& obj;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
    }

    virtual const unsigned char* getBuffer() override{
        return reinterpret_cast<const unsigned char*>(&obj);
    }
    virtual int getBufferSize() override{
        return sizeof(obj);
    }
};

template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_DIRECT_ACCESS, ObjectClass> : public SpAbstractMpiSerializer {
    const ObjectClass& obj;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
    }

    virtual const unsigned char* getBuffer() override{
        return obj.getRawData();
    }
    virtual int getBufferSize() override{
        return obj.getRawDataSize();
    }
};


template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_VEC_SERIALIZER_TYPE, ObjectClass> : public SpAbstractMpiSerializer {
    static_assert(SpGetSerializationTypeHelper::is_stdvector<typename std::decay<ObjectClass>::type>::value,
                  "Vector is expected here");

    const ObjectClass& obj;
    SpSerializer serializer;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
        serializer.append(inObj, "data");
    }

    virtual const unsigned char* getBuffer() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();
        return &buffer[0];
    }
    virtual int getBufferSize() override{
        const std::vector<unsigned char> &buffer = serializer.getBuffer();
        return buffer.size();
    }
};


template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_VEC_RAW_TYPE, ObjectClass> : public SpAbstractMpiSerializer {
    static_assert(SpGetSerializationTypeHelper::is_stdvector<typename std::decay<ObjectClass>::type>::value,
                  "Vector is expected here");
    const ObjectClass& obj;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
    }

    virtual const unsigned char* getBuffer() override{
        return reinterpret_cast<const unsigned char*>(obj.data());
    }
    virtual int getBufferSize() override{
        return sizeof(typename ObjectClass::value_type) * obj.size();
    }
};

template <class ObjectClass>
class SpMpiSerializer<SpSerializationType::SP_VEC_DIRECT_ACCESS, ObjectClass> : public SpAbstractMpiSerializer {
    static_assert(SpGetSerializationTypeHelper::is_stdvector<typename std::decay<ObjectClass>::type>::value,
                  "Vector is expected here");
    const ObjectClass& obj;
    std::vector<unsigned char> buffer;
public:
    SpMpiSerializer(const ObjectClass& inObj) : obj(inObj){
        buffer.resize(sizeof(std::size_t));
        const std::size_t nbElements = obj.size();
        std::memcpy(&buffer[0], &nbElements, sizeof(std::size_t));

        for(const auto& element : inObj){
            const std::size_t sizeElement = element.getRawDataSize();
            buffer.resize(buffer.size() + sizeof(std::size_t));
            std::memcpy(&buffer[buffer.size()-sizeof(std::size_t)], &sizeElement, sizeof(std::size_t));
            buffer.resize(buffer.size() + sizeElement);
            std::memcpy(&buffer[buffer.size() - sizeElement], element.getRawData(), sizeElement);
        }
    }

    virtual const unsigned char* getBuffer() override{
        return buffer.data();
    }
    virtual int getBufferSize() override{
        return buffer.size();
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

template <SpSerializationType type, class ObjectClass>
class SpMpiDeSerializer;


template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_SERIALIZER_TYPE, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        SpDeserializer deserializer(&buffer[0], bufferSize);
        
        obj = deserializer.restore<ObjectClass>("sp");
    }
};

template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_RAW_TYPE, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        assert(sizeof(ObjectClass) == bufferSize);
        std::memcpy(&obj, &buffer[0], bufferSize);
    }
};

template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_DIRECT_ACCESS, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        obj.restoreRawData(&buffer[0], bufferSize);
    }
};

template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_VEC_SERIALIZER_TYPE, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        SpDeserializer deserializer(&buffer[0], bufferSize);
        obj = deserializer.restore<ObjectClass>("data");
    }
};

template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_VEC_RAW_TYPE, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        assert(bufferSize%sizeof(typename ObjectClass::value_type) == 0);
        obj.resize(bufferSize/sizeof(typename ObjectClass::value_type));
        std::memcpy(&obj[0], &buffer[0], bufferSize);
    }
};

template <class ObjectClass>
class SpMpiDeSerializer<SpSerializationType::SP_VEC_DIRECT_ACCESS, ObjectClass> : public SpAbstractMpiDeSerializer {
    ObjectClass& obj;
public:
    SpMpiDeSerializer(ObjectClass& inObj) : obj(inObj){}

    void deserialize(const unsigned char* buffer, int bufferSize) override{
        std::size_t nbElements;
        std::memcpy(&nbElements, &buffer[0], sizeof(std::size_t));
        obj.resize(nbElements);
        std::size_t offset = sizeof(std::size_t);
        for(std::size_t idx = 0 ; idx < nbElements ; ++idx){
            assert(offset + sizeof(std::size_t) <= bufferSize);
            std::size_t sizeElement;
            std::memcpy(&sizeElement, &buffer[offset], sizeof(std::size_t));
            offset += sizeof(std::size_t);
            assert(offset + sizeElement <= bufferSize);
            obj[idx].restoreRawData(&buffer[offset], sizeElement);
            offset += sizeElement;
        }
    }
};

#endif // SPMPISERIALIZER_HPP
