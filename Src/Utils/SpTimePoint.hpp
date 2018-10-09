///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under MIT Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPTIMEPOINT_HPP
#define SPTIMEPOINT_HPP

#include <chrono>
#include <cmath>

class SpTimePoint {
    std::chrono::high_resolution_clock::time_point timePoint;

public:
    SpTimePoint() : timePoint(std::chrono::high_resolution_clock::now()) {}

    void setToNow(){
        timePoint = std::chrono::high_resolution_clock::now();
    }

    double differenceWith(const SpTimePoint& inOther) const{
        using double_second_time = std::chrono::duration<double, std::ratio<1, 1>>;
        return std::abs(std::chrono::duration_cast<double_second_time>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(timePoint - inOther.timePoint)).count());
    }
};

#endif
