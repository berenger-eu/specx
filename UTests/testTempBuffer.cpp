///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

#include "Buffer/SpBufferDataView.hpp"

class TestClassWithCounter{
    static std::atomic<int> NbCreated;
    static std::atomic<int> NbDeleted;

public:
    TestClassWithCounter(){
        NbCreated += 1;
    }

    TestClassWithCounter(const TestClassWithCounter&) = delete;
    TestClassWithCounter(TestClassWithCounter&&) = delete;

    TestClassWithCounter& operator=(TestClassWithCounter&&) = delete;
    TestClassWithCounter& operator=(const TestClassWithCounter&) = delete;

    ~TestClassWithCounter(){
        NbDeleted += 1;
    }

    static int GetNbCreated(){
        return NbCreated;
    }

    static int GetNbDeleted(){
        return NbDeleted;
    }
};

std::atomic<int> TestClassWithCounter::NbCreated(0);
std::atomic<int> TestClassWithCounter::NbDeleted(0);


class BufferTest : public UTester< BufferTest > {
    using Parent = UTester< BufferTest >;

    void TestBasic(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        {
            SpBufferDataView<TestClassWithCounter> testBuffer;

            TestClassWithCounter* objectPtr = nullptr;

            runtime.task(SpWrite(testBuffer.getDataDep()),
                         [&objectPtr](SpDataBuffer<TestClassWithCounter> testObject){
                objectPtr = &(*testObject);
            });
            runtime.task(SpWrite(testBuffer.getDataDep()),
                         [this,&objectPtr](SpDataBuffer<TestClassWithCounter> testObject){
                UASSERTETRUE(objectPtr == &(*testObject));
            });

            runtime.waitAllTasks();

            runtime.stopAllThreads();

            UASSERTETRUE(TestClassWithCounter::GetNbCreated() != TestClassWithCounter::GetNbDeleted());
        }
        std::cout << "TestClassWithCounter::GetNbCreated() " << TestClassWithCounter::GetNbCreated() << std::endl;
        std::cout << "TestClassWithCounter::GetNbDeleted() " << TestClassWithCounter::GetNbDeleted() << std::endl;

        UASSERTETRUE(TestClassWithCounter::GetNbCreated() == TestClassWithCounter::GetNbDeleted());
    }

    void SetTests() {
        Parent::AddTest(&BufferTest::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(BufferTest)


