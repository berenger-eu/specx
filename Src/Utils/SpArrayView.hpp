///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPARRAYVIEW_HPP
#define SPARRAYVIEW_HPP

#include <type_traits>
#include <vector>
#include <cassert>

#include "small_vector.hpp"

/**
 * An array view should be used to include/exclude itmes
 * from an array when declaring dependences.
 * Please refer to the corresponding examples/tests.
 */
class SpArrayView {
    template<class ForwardIt, class T, class Compare>
    static ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp)
    {
        ForwardIt it;
        typename std::iterator_traits<ForwardIt>::difference_type count, step;
        count = std::distance(first,last);

        while (count > 0) {
            it = first;
            step = count / 2;
            std::advance(it, step);
            if (comp(*it, value)) {
                first = ++it;
                count -= step + 1;
            }
            else
                count = step;
        }
        return first;
    }

    struct Interval {
        long int firstIdx;
        long int lastIdx;
        long int step;

        long int lastElement() const {
            return std::max(0L,((lastIdx-firstIdx-1)/step)*step) + firstIdx;
        }

        bool isInside(const long int inIdx) const{
            return firstIdx <= inIdx && inIdx < lastIdx
                    && (inIdx-firstIdx)%step == 0;
        }

        long int nbElements() const{
            return (lastIdx-firstIdx)/step;
        }
    };

    small_vector<Interval> intervals;

public:
    class Iterator{
        const SpArrayView& view;
        long int currentInterval;
        long int currentElement;

        explicit Iterator(const SpArrayView& inView, const long int inStartInterval = 0)
            : view(inView), currentInterval(inStartInterval), currentElement(0){
        }
    public:

        bool operator!=(const Iterator& other) const {
            return &this->view != &other.view || this->currentInterval != other.currentInterval
                    || this->currentElement != other.currentElement;
        }

        long int operator*() const {
            return view.intervals[currentInterval].firstIdx + currentElement;
        }
        Iterator& operator++() {
            currentElement += view.intervals[currentInterval].step;
            if(view.intervals[currentInterval].lastIdx <= view.intervals[currentInterval].firstIdx + currentElement){
                currentInterval += 1;
                currentElement = 0;
            }
            return *this;
        }

        friend SpArrayView;
    };

    SpArrayView(const long int inFirstIdx, const long int inLastIdx, const long int inStep = 1){
        if(inFirstIdx != inLastIdx){
            intervals.emplace_back(Interval{inFirstIdx, inLastIdx, inStep});
        }
    }

    SpArrayView(const long int inSize){
        if(inSize){
            intervals.emplace_back(Interval{0, inSize, 1});
        }
    }

    Iterator begin() const {
        return Iterator(*this);
    }

    Iterator end() const {
        return Iterator(*this, intervals.size());
    }

    SpArrayView& addInterval(const long int inFirstIdx, const long int inLastIdx, const long int inStep = 1){
        // TODO add with better complexity
        for(auto idx = inFirstIdx ; idx < inLastIdx ; idx += inStep){
            addItem(idx);
        }
        return *this;
    }

    SpArrayView& addItem(const long int inIdx){
        auto iter = lower_bound(intervals.begin(), intervals.end(), inIdx,
                                     [](const Interval& element, const long int& value){
            return element.lastIdx <= value;
        });
        // If the last block do not contains the idx, we are on end
        if(iter == intervals.end()){
            // If there is a block that can be extended
            if(intervals.size() && intervals.back().nbElements() == 1){
                intervals.back().lastIdx = inIdx + 1;
                intervals.back().step = inIdx - intervals.back().firstIdx;
            }
            else if(intervals.size() && intervals.back().lastElement() + intervals.back().step == inIdx ){
                intervals.back().lastIdx += intervals.back().step;
            }
            else{
                intervals.emplace_back(Interval{inIdx, inIdx+1, 1});
            }
        }
        // If already exist
        else if((*iter).isInside(inIdx)){
        }
        // If covered by an interval but not with the correct step
        else if((*iter).firstIdx <= inIdx){
            if((*iter).nbElements() == 1){
                (*iter).lastIdx = inIdx + 1;
                (*iter).step = inIdx - (*iter).firstIdx;
            }
            else if(inIdx - (*iter).firstIdx < (*iter).step){
                iter = intervals.insert(iter, Interval{(*iter).firstIdx, inIdx+1, inIdx-(*iter).firstIdx});
                ++iter;
                (*iter).firstIdx = (*iter).firstIdx + (*iter).step;
            }
            else if((*iter).lastElement() < inIdx){
                iter = intervals.insert(iter, Interval{(*iter).firstIdx, inIdx, (*iter).step});
                ++iter;
                (*iter).firstIdx = inIdx;
                (*iter).lastIdx = inIdx+1;
                (*iter).step = 1;
            }
            else{
                iter = intervals.insert(iter, Interval{(*iter).firstIdx, inIdx, (*iter).step});
                iter = intervals.insert(iter, Interval{inIdx, inIdx+1, 1});
                ++iter;
                (*iter).firstIdx = ((inIdx-(*iter).firstIdx + (*iter).step-1)/(*iter).step)*(*iter).step + (*iter).firstIdx;
            }
        }
        else if((*iter).nbElements() == 1
                || inIdx == (*iter).firstIdx - (*iter).step){
            if((*iter).nbElements() == 1){
                (*iter).step = (*iter).firstIdx - inIdx;
                (*iter).lastIdx = (*iter).firstIdx+1;
                (*iter).firstIdx = inIdx;
            }
            else{
                (*iter).firstIdx -= (*iter).step;
            }
        }
        else if(iter == intervals.begin()){
            intervals.insert(iter, Interval{inIdx, inIdx+1, 1});
        }
        else {
            --iter;
            if((*iter).nbElements() == 1){
                (*iter).step = inIdx - (*iter).firstIdx;
                (*iter).lastIdx = inIdx+1;
            }
            else if((*iter).lastElement() + (*iter).step == inIdx){
                (*iter).lastIdx += (*iter).step;
            }
            else{
                ++iter;
                intervals.insert(iter, Interval{inIdx, inIdx+1, 1});
            }
        }
        return *this;
    }

    SpArrayView& addItems(const std::initializer_list<long int> inIdxs){
        for(const long int& idx : inIdxs){
            addItem(idx);
        }
        return *this;
    }

    template <class ... Params>
    SpArrayView& addItems(Params ... inIdxs){
        addItems({inIdxs...});
        return *this;
    }

    SpArrayView& removeInterval(const long int inFirstIdx, const long int inLastIdx, const long int inStep = 1){
        // TODO remove with better complexity
        for(auto idx = inFirstIdx ; idx < inLastIdx ; idx += inStep){
            removeItem(idx);
        }
        return *this;
    }

    SpArrayView& removeItem(const long int inIdx){
        auto iter = lower_bound(intervals.begin(), intervals.end(), inIdx,
                                     [](const Interval& element, const long int& value) -> bool {
            return element.lastIdx <= value;
        });
        if(iter != intervals.end() && (*iter).isInside(inIdx)){
            assert((*iter).nbElements() != 0);
            if((*iter).firstIdx == inIdx){
                (*iter).firstIdx += (*iter).step;
                if((*iter).nbElements() == 0){
                    intervals.erase(iter);
                }
            }
            else if(inIdx == (*iter).lastElement()){
                (*iter).lastIdx -= 1;
                if((*iter).nbElements() == 0){
                    intervals.erase(iter);
                }
            }
            else{
                auto beforeIter = intervals.insert(iter, *iter);
                (*beforeIter).lastIdx = inIdx;
                (*(beforeIter+1)).firstIdx = inIdx+(*(beforeIter+1)).step;
            }
        }
        return *this;
    }

    SpArrayView& removeItems(const std::initializer_list<long int> inIdxs){
        for(const long int& idx : inIdxs){
            removeItem(idx);
        }
        return *this;
    }

    template <class ... Params>
    SpArrayView& removeItems(Params ... inIdxs){
        removeItems({inIdxs...});
        return *this;
    }

    long int getNbIntervals() const {
        return static_cast<long int>(intervals.size());
    }

    friend Iterator;
};

#endif // SPARRAYVIEW_HPP
