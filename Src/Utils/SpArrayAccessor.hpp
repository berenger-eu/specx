///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPARRAYACCESSOR_HPP
#define SPARRAYACCESSOR_HPP

#include <type_traits>
#include <vector>
#include <cassert>

template <class ObjectType>
class SpArrayAccessor {
    std::vector<ObjectType*> dataPtr;
    std::vector<long int> dataIdx;

public:
    class ConstIterator{
        const SpArrayAccessor<ObjectType>& view;
        long int currentIndex;

        explicit ConstIterator(const SpArrayAccessor<ObjectType>& inView, const long int inStartIndex = 0)
            : view(inView), currentIndex(inStartIndex){
        }
    public:
        bool operator!=(const ConstIterator& other) const {
            return &this->view != &other.view || this->currentIndex != other.currentIndex;
        }

        const ObjectType* operator*() const {
            return view.dataPtr[currentIndex];
        }

        ConstIterator& operator++() {
            currentIndex += 1;
            return *this;
        }

        friend SpArrayAccessor<ObjectType>;
    };

    class Iterator{
        SpArrayAccessor<ObjectType>& view;
        long int currentIndex;

        explicit Iterator(SpArrayAccessor<ObjectType>& inView, const long int inStartIndex = 0)
            : view(inView), currentIndex(inStartIndex){
        }
    public:
        bool operator!=(const Iterator& other) const {
            return &this->view != &other.view || this->currentIndex != other.currentIndex;
        }

        const ObjectType* operator*() const {
            return view.dataPtr[currentIndex];
        }

        ObjectType* operator*() {
            return view.dataPtr[currentIndex];
        }

        Iterator& operator++() {
            currentIndex += 1;
            return *this;
        }

        friend SpArrayAccessor<ObjectType>;
    };

    template <class VHC>
    SpArrayAccessor(ObjectType* inHandle, VHC&& inView){
        for(const long int idx : inView){
            ObjectType* ptr = &inHandle[idx];
            dataPtr.push_back(ptr);
            dataIdx.push_back(idx);
        }
    }

    SpArrayAccessor(const SpArrayAccessor&) = default;
    SpArrayAccessor(SpArrayAccessor&&) = default;
    SpArrayAccessor& operator=(const SpArrayAccessor&) = delete;
    SpArrayAccessor& operator=(SpArrayAccessor&&) = delete;

    void updatePtr(const long int position, ObjectType* ptr){
        assert(position < static_cast<long int>(dataPtr.size()));
        dataPtr[position] = ptr;
    }

    template<typename ArrayTypeSFINAE = ObjectType>
    typename std::enable_if<!std::is_const<ArrayTypeSFINAE>::value, ArrayTypeSFINAE&>::type
    getAt(const long int inIndex){
        return *dataPtr[inIndex];
    }

    const ObjectType& getAt(const long int inIndex) const{
        return *dataPtr[inIndex];
    }

    long int getIndexAt(const long int inIndex) const{
        return dataIdx[inIndex];
    }

    long int getSize() const{
        return static_cast<long int>(dataPtr.size());
    }

    ConstIterator begin() const {
        return ConstIterator(*this);
    }

    ConstIterator end() const {
        return ConstIterator(*this, dataPtr.size());
    }

    template<typename ArrayTypeSFINAE = ObjectType>
    typename std::enable_if<!std::is_const<ArrayTypeSFINAE>::value, Iterator>::type
    begin() {
        return Iterator(*this);
    }

    template<typename ArrayTypeSFINAE = ObjectType>
    typename std::enable_if<!std::is_const<ArrayTypeSFINAE>::value, Iterator>::type
    end() {
        return Iterator(*this, dataPtr.size());
    }
};

#endif // SPARRAYVIEW_HPP
