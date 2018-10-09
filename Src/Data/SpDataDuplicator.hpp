///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under MIT Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDATADUPLICATOR_HPP
#define SPDATADUPLICATOR_HPP

#include <memory>

class SpAbstractDataDuplicator {
public:
    virtual ~SpAbstractDataDuplicator(){}
    virtual void* duplicateData(void*) const = 0;
    virtual void deleteDuplicata(void*) const = 0;
    virtual std::unique_ptr<SpAbstractDataDuplicator> clone() const = 0;
};

enum SpTargetDuplicateMode{
    DUPLICATE_COPY_CST,
    DUPLICATE_CLONE,
    NO_DUPLICATE
};


#include <type_traits>

template <class C>
class SpDataHasCloneFunction
{
    template <class T>
    static std::true_type testSignature(void* (T::*)() const);

    template <class T>
    static decltype(testSignature(&T::clone)) test(std::nullptr_t);

    template <class T>
    static std::false_type test(...);

public:
    using type = decltype(test<C>(nullptr));
    static const bool value = type::value;
};

template <class DataType>
struct SpDataCanBeDuplicate{
    // TODO here it is no more clone that we use
    static const bool value = (std::is_copy_constructible<DataType>::value || SpDataHasCloneFunction<DataType>::value);
};

template <class DataType>
struct SpDuplicateSelector{
    static const SpTargetDuplicateMode mode = (std::is_copy_constructible<DataType>::value?DUPLICATE_COPY_CST:
                SpDataHasCloneFunction<DataType>::value?DUPLICATE_CLONE:NO_DUPLICATE);
};


template <class DataType, SpTargetDuplicateMode CpConst = SpDuplicateSelector<DataType>::mode >
class SpDataDuplicator;

template <class DataType>
class SpDataDuplicator<DataType, DUPLICATE_COPY_CST> : public SpAbstractDataDuplicator{
public:
    void* duplicateData(void* inPtr) const final{
        return new DataType(*reinterpret_cast<DataType*>(inPtr));
    }

    void deleteDuplicata(void* inPtr) const final{
        delete reinterpret_cast<DataType*>(inPtr);
    }

    std::unique_ptr<SpAbstractDataDuplicator> clone() const final{
        return std::unique_ptr<SpAbstractDataDuplicator>(new SpDataDuplicator<DataType, DUPLICATE_COPY_CST>());
    }
};

template <class DataType>
class SpDataDuplicator<DataType, DUPLICATE_CLONE> : public SpAbstractDataDuplicator{
public:
    void* duplicateData(void* inPtr) const final{
        return reinterpret_cast<DataType*>(inPtr)->clone();
    }

    void deleteDuplicata(void* inPtr) const final{
        delete reinterpret_cast<DataType*>(inPtr);
    }

    std::unique_ptr<SpAbstractDataDuplicator> clone() const final{
        return std::unique_ptr<SpAbstractDataDuplicator>(new SpDataDuplicator<DataType, DUPLICATE_CLONE>());
    }
};

template <class DataType>
class SpDataDuplicator<DataType, NO_DUPLICATE> : public SpAbstractDataDuplicator{
public:
    void* duplicateData(void* inPtr) const final{
        return inPtr;
    }

    void deleteDuplicata(void*) const final{
    }

    std::unique_ptr<SpAbstractDataDuplicator> clone() const final{
        return std::unique_ptr<SpAbstractDataDuplicator>(new SpDataDuplicator<DataType, NO_DUPLICATE>());
    }
};


#endif
