///////////////////////////////////////////////////////////////////////////
// Thomas Millot (c), Unistra, 2020
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPPHILOXGENERATOR_HPP
#define SPPHILOXGENERATOR_HPP
#include <iostream>
#include <random>
#include <array>

// Implementation of the Philox algorithm to generate random numbers in parallel.
// http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
// Also based on an implementation of the algorithm by Tensorflow.
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/random/philox_random.h
// It's the Philox-4×32 version, meaning 4 32bits random numbers each time.
// The number of cycles can be user defined, by default it's 10.
// The engine satisfies C++ named requirements RandomNumberDistribution,
// so it can be used with std::uniform_real_distribution for example.

// Exactly the same interface as SpMTGenerator, so it can be interchangeable
template<class RealType = double>
class SpPhiloxGenerator {
    class philox4x32 {
        typedef uint_fast32_t uint32;
        typedef uint_fast64_t uint64;

        static constexpr int DEFAULT_CYCLES = 10;
    public:

        // An array of four uint32, the results of the philox4 engine
        using Result = std::array<uint32, 4>;

        // 64-bit seed stored in two uint32
        using Key = std::array<uint32, 2>;

        philox4x32() = default;

        explicit philox4x32(uint64 seed, int cycles = DEFAULT_CYCLES)
        : counter_(), temp_results_(), key_(), temp_counter_(0), cycles_(cycles),
        force_computation_(true), operatorPPcounter(0)
        {
            // Splitting the seed in two
            key_[0] = static_cast<uint32>(seed);
            key_[1] = static_cast<uint32>(seed >> 32);

            counter_.fill(0);
            temp_results_.fill(0);
        }

        // Returns the minimum value productible by the engine
        static constexpr uint32 min() { return _Min; }

        // Returns the maximum value productible by the engine
        static constexpr uint32 max() { return _Max; }

        // Skip the specified number of steps
        void Skip(uint64 count) {
            if(count > 0) {
                
                const auto nbStepsToNextMultipleOf4 = 4 - temp_counter_;
                
                if(count <= nbStepsToNextMultipleOf4) {
                    temp_counter_ += count;
                    return; 
                }
                
                count -= nbStepsToNextMultipleOf4;
                
                const auto nbOfCounterIncrements = count / 4 + 1;
                temp_counter_ = count % 4;
                
                const auto count_lo = static_cast<uint32>(nbOfCounterIncrements);
                auto count_hi = static_cast<uint32>(nbOfCounterIncrements >> 32);
                
                // 128 bit add
                
                counter_[0] += count_lo;
                if (counter_[0] < count_lo) {
                    ++count_hi;
                }

                counter_[1] += count_hi;
                if (counter_[1] < count_hi) {
                    if (++counter_[2] == 0) {
                        ++counter_[3];
                    }
                }
                
                force_computation_ = true;
            }
        }

        // Returns a random number using the philox engine
        uint32 operator()() {
            operatorPPcounter++;

            if(temp_counter_ == 4) {
                temp_counter_ = 0;
                SkipOne();
                force_computation_ = true;
            }
            
            if(force_computation_) {
                force_computation_ = false;
                temp_results_ = counter_;
                ExecuteRounds();
            }

            uint32 value = temp_results_[temp_counter_];
            temp_counter_++;

            return value;
        }

        auto getOperatorPPCounter() const{
            return operatorPPcounter;
        }

    private:

        // Using the same constants as recommended in the original paper.
        static constexpr uint32 kPhiloxW32A = 0x9E3779B9;
        static constexpr uint32 kPhiloxW32B = 0xBB67AE85;
        static constexpr uint32 kPhiloxM4x32A = 0xD2511F53;
        static constexpr uint32 kPhiloxM4x32B = 0xCD9E8D57;

        // The minimum return value
        static constexpr uint32 _Min = 0;
        // The maximum return value
        static constexpr uint32 _Max = UINT_FAST32_MAX;

        // The counter for the current state of the engine
        Result counter_;

        // Keeping the last to results to improve performances during consecutive call
        Result temp_results_;

        // The split seed
        Key key_;

        // To iterate through the temp_results_
        uint64 temp_counter_;

        // The number of cycles used to generate randomness
        int cycles_;

        // To force the engine to compute the rounds to populates temp_results_
        bool force_computation_;

        // The number of times operator () is called to ensure that the STL
        // always call it once
        uint32 operatorPPcounter;

        // Skip one step
        void SkipOne() {
            // 128 bit increment
            if (++counter_[0] == 0) {
                if (++counter_[1] == 0) {
                    if (++counter_[2] == 0) {
                        ++counter_[3];
                    }
                }
            }
        }

        // Helper function to return the lower and higher 32-bits from two 32-bit integer multiplications.
        static void MultiplyHighLow(uint32 a, uint32 b, uint32 *result_low, uint32 *result_high) {

            const uint64 product = static_cast<uint64>(a) * b;
            *result_low = static_cast<uint32>(product);
            *result_high = static_cast<uint32>(product >> 32);

        }

        void ExecuteRounds() {

            Key key = key_;

            // Run the single rounds for ten times.
            for (int i = 0; i < cycles_; ++i) {
                temp_results_ = ComputeSingleRound(temp_results_, key);
                RaiseKey(&key);
            }
        }

        // Helper function for a single round of the underlying Philox algorithm.
        static Result ComputeSingleRound(const Result &counter, const Key &key) {
            uint32 lo0;
            uint32 hi0;
            MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

            uint32 lo1;
            uint32 hi1;
            MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

            Result result;
            result[0] = hi1 ^ counter[1] ^ key[0];
            result[1] = lo1;
            result[2] = hi0 ^ counter[3] ^ key[1];
            result[3] = lo0;
            return result;
        }

        void RaiseKey(Key *key) {
            (*key)[0] += kPhiloxW32A;
            (*key)[1] += kPhiloxW32B;
        }
    };

    philox4x32 phEngine;
    std::uniform_real_distribution<RealType> dis01;
    std::size_t nbValuesGenerated;

public:
    explicit SpPhiloxGenerator() : phEngine(std::random_device()()), dis01(0, 1), nbValuesGenerated(0) {}

    explicit SpPhiloxGenerator(const size_t inSeed) : phEngine(inSeed), dis01(0, 1), nbValuesGenerated(0) {}

    SpPhiloxGenerator(const SpPhiloxGenerator &) = default;

    SpPhiloxGenerator(SpPhiloxGenerator &&) = default;

    SpPhiloxGenerator &operator=(const SpPhiloxGenerator &) = default;

    SpPhiloxGenerator &operator=(SpPhiloxGenerator &&) = default;

    SpPhiloxGenerator &skip(const size_t inNbToSkip) {
        if(inNbToSkip == 0){
            return *this;
        }

        phEngine.Skip(inNbToSkip);
        
        nbValuesGenerated += inNbToSkip;
        
        return *this;
    }

    RealType getRand01() {
        nbValuesGenerated++;
        [[maybe_unused]] const auto counterOperatorPPBefore = phEngine.getOperatorPPCounter();
        const RealType number = dis01(phEngine);
        [[maybe_unused]] const auto counterOperatorPPAfter = phEngine.getOperatorPPCounter();
        assert(counterOperatorPPAfter == counterOperatorPPBefore+1);
        return number;
    }

    size_t getNbValuesGenerated() const {
        return nbValuesGenerated;
    }
};


#endif
