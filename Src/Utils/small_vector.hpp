/*
 * Implementation of a vector optimized for the case where only a small number of elements
 * are stored.
 * The vector starts out storing elements in an inline storage buffer of size N. If the vector size
 * grows beyond N elements, the vector stores the elements in a dynamically allocated buffer.
 * The implementation of small_vector is largely inspired by the SmallVector class implementation from 
 * the LLVM framework codebase.
 * Link to the implementation of SmallVector in the LLVM codebase :
 * https://github.com/llvm/llvm-project/blob/master/llvm/include/llvm/ADT/SmallVector.h
 * The implementation of the function nextPowerOfTwo has been largely inspired by the implementation of
 * the function NextPowerOf2 from the LLVM codebase :
 * Link to the implementation of NextPowerOf2 in the LLVM codebase:
 * https://github.com/llvm/llvm-project/blob/master/llvm/include/llvm/Support/MathExtras.h
 */

#ifndef SMALL_VECTOR_HPP
#define SMALL_VECTOR_HPP

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <algorithm>

inline uint64_t nextPowerOfTwo(uint64_t n);

class small_vector_common_base {
protected:
    size_t capacity_;
    size_t originalCapacity_;
    size_t size_;
    void *beginPtr_;
    
    small_vector_common_base() = delete;
    small_vector_common_base(size_t originalCapacity, void *beginPtr)
        : capacity_(originalCapacity), originalCapacity_(originalCapacity), size_(0), beginPtr_(beginPtr) {}
};

template <typename T>
struct small_vector_internal_storage_offset_computation_structure {
    alignas(small_vector_common_base) char base[sizeof(small_vector_common_base)];
    alignas(T) char firstElt[sizeof(T)];
};

template <typename T>
class small_vector_base : public small_vector_common_base {
public:

    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using value_type = T;
    using iterator = T *;
    using const_iterator = const T *;
    
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    
    using reference =  T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

private:
     void grow_capacity_to_a_minimum_of(size_type minCapacity) {
        assert(minCapacity <= UINT32_MAX);
        
        size_type newCapacity = std::min(std::max(minCapacity, size_t(nextPowerOfTwo(capacity()))), size_t(UINT32_MAX));
        
        void *p = std::malloc(newCapacity*sizeof(T));
        
        std::uninitialized_copy(std::move_iterator<iterator>(begin()), std::move_iterator<iterator>(end()), static_cast<iterator>(p));
        
        std::destroy(rbegin(), rend());
        
        if(!is_small()) {
            std::free(beginPtr_);
        }
        
        beginPtr_ = p;
        capacity_ = newCapacity;
    }

protected:
    small_vector_base() = delete;
    small_vector_base(size_t originalCapacity)
        : small_vector_common_base(originalCapacity, getPointerToInternalStorage()) {}

public:
    template<typename InputItTy,
             typename = typename std::enable_if<std::is_convertible<
             typename std::iterator_traits<InputItTy>::iterator_category,
             std::input_iterator_tag>::value>::type>
    void assign(InputItTy first, InputItTy last) {
        size_t nbEltsInRange = std::distance(first, last);
        
        if(capacity() >= nbEltsInRange) {
            if(nbEltsInRange <= size()) {
                std::copy(first, last, begin());
                
                if(nbEltsInRange < size()) {
                    std::destroy(rbegin(), reverse_iterator(begin() + nbEltsInRange));
                }
            }else {
                iterator i2 = begin();
                InputItTy i = first;
                
                for(size_t nb = size(); nb > 0; --nb, ++i, ++i2) {
                    *i2 = *i;
                }
                
                for(; i != last; ++i2, ++i) {
                    new (static_cast<void *>(std::addressof(*i2))) T(*i);
                }
            }
        }else {
            clear();
            grow_capacity_to_a_minimum_of(nbEltsInRange);
            std::uninitialized_copy(first, last, begin());
        }
        
        set_size(nbEltsInRange);
    }
    
    void assign(size_type count, const T &value) {
        
        if(capacity() >= count) {
            if(count <= size()) {
                std::fill_n(begin(), count, value);
                
                if(count < size()) {
                    std::destroy(rbegin(), reverse_iterator(begin() + count));
                }
            }else {
                std::fill_n(begin(), size(), value);
                std::uninitialized_fill(end(), begin() + count, value);
            }
            
        } else {
            clear();
            grow_capacity_to_a_minimum_of(count);
            std::uninitialized_fill_n(begin(), count, value);
        }
        
        set_size(count);
    }
    
    void assign(std::initializer_list<T> il) {
        assign(il.begin(), il.end());
    }
    
    small_vector_base &operator=(const small_vector_base &rhs) {
        if(this == std::addressof(rhs)) {
            return *this;
        }
        
        size_type rhsSize = rhs.size();
        
        if(size() >= rhsSize) {
            if(rhsSize > 0) {
                std::copy(rhs.begin(), rhs.end(), begin());
            }
            
            std::destroy(rbegin(), reverse_iterator(begin() + rhsSize));
            
            set_size(rhsSize);
            return *this;
        }
        
        if(capacity() < rhsSize) {
            clear();
            grow_capacity_to_a_minimum_of(rhsSize);
        }else if(size() > 0) {
            std::copy(rhs.begin(), rhs.begin() + size(), begin());
        }
        
        std::uninitialized_copy(rhs.begin() + size(), rhs.end(), begin() + size());
        set_size(rhsSize);
        return *this;
    }
    
    small_vector_base &operator=(small_vector_base &&rhs) {
        if(this == std::addressof(rhs)) {
            return *this;
        }
        
        if(!rhs.is_small()) {
            clear();
            
            if(!is_small()) {
                std::free(beginPtr_);
            }
            
            beginPtr_ = rhs.begin();
            size_ = rhs.size();
            capacity_ = rhs.capacity();
            rhs.reset_to_small();
            
            return *this;
        }
        
        size_type rhsSize = rhs.size();
        
        if(size() >= rhsSize) {
            if(rhsSize > 0) {
                std::move(rhs.begin(), rhs.end(), begin());
            }
            
            std::destroy(rbegin(), reverse_iterator(begin() + rhsSize));
            set_size(rhsSize);
            rhs.clear();
            return *this;
        }
        
        if(capacity() < rhsSize) {
            clear();
            grow_capacity_to_a_minimum_of(rhsSize);
        }else if(size() > 0) {
            std::move(rhs.begin(), rhs.begin() + size(), begin());
        }
        
        std::uninitialized_copy(std::move_iterator<iterator>(rhs.begin() + size()),
                                std::move_iterator<iterator>(rhs.end()),
                                begin() + size());
        set_size(rhsSize);
        rhs.clear();
        return *this;
    }
    
    reference at(size_type index) {
        if(index < size()) {
            return begin()[index];
        } else {
            throw std::out_of_range("small_vector::at(size_type index) std::out_or_range exception");
        }
    }
    
    reference operator[](size_type index) {
        return begin()[index];
    }
    
    const_reference operator[](size_type index) const {
        return begin()[index];
    }
    
    reference front() {
        assert(!empty());
        return begin()[0];
    }
    
    const_reference front() const {
        assert(!empty());
        return begin()[0];
    }
    
    reference back() {
        assert(!empty());
        return end()[-1];
    }
    
    const_reference back() const {
        assert(!empty());
        return end()[-1];
    }
    
    pointer data() {
        return pointer(begin());
    }
    
    const_pointer data() const { 
        return const_pointer(begin());
    }
    
    iterator begin() { return static_cast<iterator>(beginPtr_); }
    const_iterator begin() const { return static_cast<const_iterator>(beginPtr_); }
    const_iterator cbegin() const { return static_cast<const_iterator>(beginPtr_); }
    
    iterator end() { return begin() + size(); }
    const_iterator end() const { return begin() + size(); }
    const_iterator cend() const { return begin() + size(); }
    
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    const_reverse_iterator rbegin() const { return  const_reverse_iterator(end()); }
    const_reverse_iterator crbegin() const { return  const_reverse_iterator(end()); }
    
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator rend() const { return reverse_iterator(begin()); }
    const_reverse_iterator crend() const { return reverse_iterator(begin()); }
    
    bool empty() const {
        return size_ == 0;
    }
    
    size_type size() const {
        return size_;
    }
    
    size_type max_size() const {
        return size_type(-1) / sizeof(T);
    }
    
    void reserve(size_type capacity) {
        if(capacity > capacity_) {
            grow_capacity_to_a_minimum_of(capacity);
        }
    }
    
    size_type capacity() const {
        return capacity_;
    }
    
    void set_size(size_type size) {
        assert(size <= capacity());
        size_ = size;
    }
    
    void clear() {
        std::destroy(rbegin(), rend());
        set_size(0);
    }
    
    iterator insert(const_iterator pos, T &&value) {
        iterator it = const_cast<iterator>(pos);
        if(it == end()) {
            push_back(value);
            return end()-1;
        }
        
        assert(it >= begin() && it <= end() && "small_vector::insert iterator is out of bounds.");
        
        if(size() == capacity()) {
            size_t index = std::distance(begin(), it);
            grow_capacity_to_a_minimum_of(size()+1);
            it = begin() + index;
        }
        
        new (static_cast<void *>(end())) T(std::move(back()));
        std::move_backward(it, end()-1, end());
        
        const_iterator itElt = std::addressof(value);
        
        if(itElt >= it && itElt < end()) {
            ++itElt;
        }
        
        *it = std::move(*itElt);
        
        set_size(size()+1);
        
        return it;
    }
    
    iterator insert(const_iterator pos, const T &value) {
        iterator it = const_cast<iterator>(pos);
        if(it == end()) {
            push_back(value);
            return end()-1;
        }
        
        assert(it >= begin() && it <= end() && "small_vector::insert iterator is out of bounds.");
        
        if(size() == capacity()) {
            size_t index = std::distance(begin(), it);
            grow_capacity_to_a_minimum_of(size()+1);
            it = begin() + index;
        }
        
        new (static_cast<void *>(end())) T(std::move(back()));
        std::move_backward(it, end()-1, end());
        
        const_iterator itElt = std::addressof(value);
        
        if(itElt >= it && itElt < end()) {
            ++itElt;
        }
        
        *it = *itElt;
        
        set_size(size()+1);
        
        return it;
    }
    
    iterator insert(const_iterator pos, size_type count, const T &value) {
        iterator it = const_cast<iterator>(pos);
        size_t index = std::distance(begin(), it);
        
        if(it == end()) {
            reserve(size() + count);
            std::uninitialized_fill_n(end(), count, value);
            set_size(size()+count);
            return begin() + index;
        }
        
        assert(it >= begin() && it <= end() && "small_vector::insert iterator is out of bounds.");
        
        reserve(size() + count);
        
        it = begin() + index;
        
        size_t nbEltsInRange = std::distance(it, end());
        
        if(count <= nbEltsInRange) {
            std::uninitialized_copy(std::move_iterator<iterator>(end() - count),
                                    std::move_iterator<iterator>(end()), end());
            std::move_backward(it, end() - count, end());
            set_size(size()+count);
            std::fill_n(it, count, value);
            return it;
        }
        
        iterator oldEnd = end();
        set_size(size() + count);
        size_t nbOverwrittenElts = std::distance(it, oldEnd);
        std::uninitialized_copy(std::move_iterator<iterator>(it), std::move_iterator<iterator>(oldEnd), end() - nbOverwrittenElts);
        
        std::fill_n(it, nbOverwrittenElts, value);
        
        std::uninitialized_fill_n(oldEnd, count - nbOverwrittenElts, value);
        
        return it;
    }
    
    template<typename InputItTy,
             typename = typename std::enable_if<std::is_convertible<
             typename std::iterator_traits<InputItTy>::iterator_category,
             std::input_iterator_tag>::value>::type>
    iterator insert(const_iterator pos, InputItTy first, InputItTy last) {
        iterator it = const_cast<iterator>(pos);
        
        size_t index = std::distance(begin(), it);
        size_t nbEltsInInputRange = std::distance(first, last);
        
        if(it == end()) {
            reserve(size() + nbEltsInInputRange);
            std::uninitialized_copy(first, last, end());
            set_size(size() + nbEltsInInputRange);
            return begin() + index;
        }
        
        assert(it >= begin() && it <= end() && "small_vector::insert iterator is out of bounds.");
        
        reserve(size() + nbEltsInInputRange);
        
        it = begin() + index;
        
        size_t nbEltsInRangeItToEnd = std::distance(it, end());
        
        if(nbEltsInInputRange <= nbEltsInRangeItToEnd) {
            std::uninitialized_copy(std::move_iterator<iterator>(end() - nbEltsInInputRange),
                                    std::move_iterator<iterator>(end()), end());
            std::move_backward(it, end() - nbEltsInInputRange, end());
            std::copy(first, last, it);
            set_size(size() + nbEltsInInputRange);
            return it;
        }
        
        size_t rangeSizeDifference = nbEltsInInputRange - nbEltsInRangeItToEnd;
        
        std::uninitialized_move(it, end(), end() + rangeSizeDifference);
        
        for(iterator i = it; i != it + nbEltsInRangeItToEnd; ++it, ++first) {
            *i = *first;
        }
        
        std::uninitialized_copy(first, last, end());
        
        set_size(size() + nbEltsInInputRange);
        
        return it;
    }
    
    void push_back(const T &value) {
        if(size() >= capacity()) {
            grow_capacity_to_a_minimum_of(capacity()+1);
        }
        
        new (static_cast<void *>(end())) T(value);
        
        set_size(size()+1);
    }
    
    iterator erase(const_iterator pos) {
        iterator it = const_cast<iterator>(pos);

        assert(it >= this->begin() && it < this->end() && "small_vector::erase iterator to erase from is out of bounds.");
        
        std::move(it+1, end(), it);
        pop_back();
        
        return it;
    }
    
    iterator erase(const_iterator first, const_iterator last) {
        iterator itFirst = const_cast<iterator>(first);
        iterator itLast = const_cast<iterator>(last);
        
        assert(itFirst <= itLast && "small_vector::erase trying to erase invalid range.");
        assert(itFirst >= this->begin() && itLast <= this->end() && "small_vector::erase range to erase is out of bounds.");

        iterator it = std::move(itLast, end(), itFirst);
        std::destroy(rbegin(), reverse_iterator(it));
        this->set_size(std::distance(begin(), it));
        
        return itFirst;
    }
    
    void push_back(T &&value) {
        if(size() >= capacity()) {
            grow_capacity_to_a_minimum_of(capacity()+1);
        }
        
        new (static_cast<void *>(end())) T(std::move(value));
        
        set_size(size()+1);
    }
    
    template <typename... Args>
    reference emplace_back(Args &&... args) {
        if(size() >= capacity()) {
            grow_capacity_to_a_minimum_of(capacity()+1);
        }
        
        new (static_cast<void *>(end())) T(std::forward<Args>(args)...);
        
        set_size(size()+1);
        
        return back();
    }
    
    void pop_back() {
        assert(!empty());
        set_size(size()-1);
        end()->~T();
    }
    
    void resize(size_type count) {
        if(count < size()) {
            std::destroy(rbegin(), reverse_iterator(begin() + count));
        }else if(count > size()) {
            if(capacity() < count) {
                grow_capacity_to_a_minimum_of(count);
            }
            std::uninitialized_value_construct(end(), begin() + count);
        }
        set_size(count);
    }
    
    void resize(size_type count, const value_type &value) {
        if(count < size()) {
            std::destroy(rbegin(), reverse_iterator(begin() + count));
        }else if(count > size()) {
            if(capacity() < count) {
                grow_capacity_to_a_minimum_of(count);
            }
            std::uninitialized_fill(end(), begin() + count, value);
        }
        set_size(count);
    }
    
    void swap(small_vector_base &rhs) {
        if(this == std::addressof(rhs)) {
            return;
        }
        
        if(!is_small() && !rhs.is_small()) {
            std::swap(capacity_, rhs.capacity_);
            std::swap(size_, rhs.size_);
            std::swap(beginPtr_, rhs.beginPtr_);
            return;
        }
        
        if(rhs.capacity() < size()) {
            rhs.grow_capacity_to_a_minimum_of(size());
        }
        
        if(capacity() < rhs.size()) {
            grow_capacity_to_a_minimum_of(rhs.size());
        }
        
        size_type minSize = std::min(size(), rhs.size());
        
        std::swap_ranges(begin(), begin() + minSize, rhs.begin()); 
        
        if(size() > rhs.size()) {
            size_type nbEltsDiff = size() - rhs.size();
            std::uninitialized_copy(begin() + minSize, end(), rhs.end());
            rhs.set_size(rhs.size() + nbEltsDiff);
            set_size(minSize);
        }else if(rhs.size() > size()) {
            size_type nbEltsDiff = rhs.size() - size();
            std::uninitialized_copy(rhs.begin() + minSize, rhs.end(), end());
            set_size(size() + nbEltsDiff);
            rhs.set_size(minSize);
        }
    }
    
    bool operator==(const small_vector_base &rhs) {
        return size() == rhs.size() && std::equal(begin(), end(), rhs.begin());
    }
    
    bool is_small() const {
        return beginPtr_ == getPointerToInternalStorage();
    }
    
    void *getPointerToInternalStorage() const {
        return const_cast<void *>(reinterpret_cast<const void *>(reinterpret_cast<const char *>(this) +
               offsetof(small_vector_internal_storage_offset_computation_structure<T>, firstElt)));
    }
    
    void reset_to_small() {
        beginPtr_ = getPointerToInternalStorage();
        capacity_ = originalCapacity_;
        size_ = 0;
    }
};

template <typename T, size_t N>
struct small_vector_storage {
    alignas(T) char smallBuffer_[sizeof(T)*N];
};

template <typename T>
struct alignas(T) small_vector_storage<T, 0> { };

template <typename T, size_t N = 64>
class small_vector : public small_vector_base<T>, public small_vector_storage<T, N> {
public:
    using typename small_vector_base<T>::size_type;
    using typename small_vector_base<T>::difference_type;
    using typename small_vector_base<T>::value_type;
    using typename small_vector_base<T>::iterator;
    using typename small_vector_base<T>::const_iterator;
    
    using typename small_vector_base<T>::reverse_iterator;
    using typename small_vector_base<T>::const_reverse_iterator;
    
    using typename small_vector_base<T>::reference;
    using typename small_vector_base<T>::const_reference;
    using typename small_vector_base<T>::pointer;
    using typename small_vector_base<T>::const_pointer;
    
    small_vector() : small_vector_base<T>(N) {}
    
    explicit small_vector(size_t count, const T &value = T()) : small_vector() {
        this->assign(count, value);
    }
    
    small_vector(const small_vector &rhs) : small_vector() {
        if(!rhs.empty()) {
            small_vector_base<T>::operator=(rhs);
        }
    }
    
    small_vector(small_vector &&rhs) : small_vector() {
        if(!rhs.empty()) {
            small_vector_base<T>::operator=(std::move(rhs));
        }
    }
    
    template<typename InputItTy,
             typename = typename std::enable_if<std::is_convertible<
             typename std::iterator_traits<InputItTy>::iterator_category,
             std::input_iterator_tag>::value>::type>
    small_vector(InputItTy first, InputItTy last) : small_vector() {
        this->assign(first, last);
    }
    
    small_vector(std::initializer_list<T> il) : small_vector() {
        this->assign(il);
    }
    
    ~small_vector() {
        std::destroy(this->rbegin(), this->rend());
        if(!this->is_small()) {
            std::free(this->begin());
        }
    }
    
    small_vector &operator=(const small_vector &rhs) {
        small_vector_base<T>::operator=(rhs);
        return *this;
    }
    
    small_vector &operator=(const small_vector_base<T> &rhs) {
        small_vector_base<T>::operator=(rhs);
        return *this;
    }
    
    small_vector &operator=(small_vector &&rhs) {
        small_vector_base<T>::operator=(std::move(rhs));
        return *this;
    }
    
    small_vector &operator=(small_vector_base<T> &&rhs) {
        small_vector_base<T>::operator=(std::move(rhs));
        return *this;
    }
    
    small_vector &operator=(std::initializer_list<T> il) {
        this->assign(il);
        return *this;
    }
};

inline uint64_t nextPowerOfTwo(uint64_t n) {
    n |= (n >> 1);
    n |= (n >> 2);
    n |= (n >> 4);
    n |= (n >> 8);
    n |= (n >> 16);
    n |= (n >> 32);
    
    return n + 1;
}

namespace std {
    template <typename T, size_t N>
    inline void swap(small_vector<T, N> &lhs, small_vector<T, N> &rhs) {
        lhs.swap(rhs);
    }
    
    template <typename T>
    inline void swap(small_vector_base<T> &lhs, small_vector_base<T> &rhs) {
        lhs.swap(rhs);
    }
}

template <typename T>
struct is_instantiation_of_small_vector : std::false_type {};

template <typename T, size_t N>
struct is_instantiation_of_small_vector<small_vector<T, N>> : std::true_type {};

#endif
