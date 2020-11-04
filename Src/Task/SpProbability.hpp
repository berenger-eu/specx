///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPPROBABILITY_HPP
#define SPPROBABILITY_HPP

#include <cassert>

/**
 * This class should be used to inform the runtime
 * about the propability that a potential task will
 * modify its target data.
 */
class SpProbability{
    bool activated;
    double probability;
    int includedProbabilities;
public:
    explicit SpProbability(const double inProbability)
        : activated(true), probability(inProbability), includedProbabilities(1){
        assert(0 <= probability && probability <= 1);
    }

    explicit SpProbability()
        : activated(false), probability(0), includedProbabilities(0){
        assert(0 <= probability && probability <= 1);
    }

    double getProbability() const{
        return probability;
    }

    bool isUsed() const{
        return activated;
    }

    void append(const SpProbability& other){
        if(activated == false){
            (*this) = other;
        }
        else if(other.activated == true){
            probability = (probability*includedProbabilities + other.probability)/includedProbabilities;
            includedProbabilities += 1;
        }
    }
};

#endif

