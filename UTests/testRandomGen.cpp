#include "Utils/SpModes.hpp"
#include "Random/SpMTGenerator.hpp"
#include "Random/SpPhiloxGenerator.hpp"

#include "UTester.hpp"

class TestRandomGen : public UTester< TestRandomGen > {
    using Parent = UTester< TestRandomGen >;

    using TestType = double;

    template <class RandomGenType>
    void TestBasic(){
        const size_t testSize = 10000;
        std::vector<TestType> buffer(testSize);
        for(const size_t seed : std::vector<size_t>{{0, 1, 34, 534534}}){
            RandomGenType generator(seed);

            for(size_t idx = 0 ; idx < testSize ; ++idx){
                buffer[idx] = generator.getRand01();
            }

            {
                RandomGenType generatorTest(seed);

                for(size_t idx = 0 ; idx < testSize ; ++idx){
                    UASSERTEEQUAL(buffer[idx], generatorTest.getRand01());
                }
            }

            for(size_t idxSkip = 1 ; idxSkip < 100 ; ++idxSkip){
                RandomGenType generatorTest(seed);

                for(size_t idx = 0 ; idx < testSize ; ++idx){
                    if(idx % idxSkip == 0){
                        UASSERTEEQUAL(buffer[idx], generatorTest.getRand01());
                        generatorTest.skip(idxSkip-1);
                    }
                }
            }
        }
    }

    void SetTests() {
        Parent::AddTest(&TestRandomGen::TestBasic<SpMTGenerator<TestType>>, "Basic test for SpMTGenerator");
        Parent::AddTest(&TestRandomGen::TestBasic<SpPhiloxGenerator<TestType>>, "Basic test for SpPhiloxGenerator");
    }
};

// You must do this
TestClass(TestRandomGen)

