///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorker.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Utils/SpBlockTuple.hpp"
#include "Utils/SpArrayBlock.hpp"
#include "Utils/SpAlignment.hpp"

class BlockTupleTest : public UTester< BlockTupleTest > {
    using Parent = UTester< BlockTupleTest >;

    void Test(){
        SpBlockTuple<SpArrayBlock<int>, SpArrayBlock<float>> bt({40, 2});
        SpTaskGraph tg;
        
        tg.task(SpWrite(bt), SpGpu([](auto btParam) {}));
    }

    void SetTests() {
        Parent::AddTest(&BlockTupleTest::Test, "Basic block tuple test");
    }
};

// You must do this
TestClass(BlockTupleTest)
