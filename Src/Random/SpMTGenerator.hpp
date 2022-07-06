///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPMTGENERATOR_HPP
#define SPMTGENERATOR_HPP

#include <random>

/**
 * The is a random generator based on mt19937.
 * It supports skipping values, but it is simply an emulation,
 * because to skip N values, the generator calls N times rand.
 * Therefore, it is adequate for testing but should not be used
 * when performance is important.
 */
template <class RealType = double>
class SpMTGenerator {
    std::mt19937_64 mtEngine;
    std::uniform_real_distribution<RealType> dis01;
    std::size_t nbValuesGenerated;

public:
    explicit SpMTGenerator() : mtEngine(std::random_device()()), dis01(0,1), nbValuesGenerated(0){}

    explicit SpMTGenerator(const size_t inSeed) : mtEngine(inSeed), dis01(0,1), nbValuesGenerated(0){}

    SpMTGenerator(const SpMTGenerator&) = default;
    SpMTGenerator(SpMTGenerator&&) = default;
    SpMTGenerator& operator=(const SpMTGenerator&) = default;
    SpMTGenerator& operator=(SpMTGenerator&&) = default;

    SpMTGenerator& skip(const size_t inNbToSeep){
        // To skip a value, we call rand() and do not use
        // the result of the call.
        for(size_t idx = 0 ; idx < inNbToSeep ; ++idx){
            getRand01();
        }
        return *this;
    }

    RealType getRand01(){
        nbValuesGenerated += 1;
        return dis01(mtEngine);
    }

    size_t getNbValuesGenerated() const{
        return nbValuesGenerated;
    }
};

#endif
