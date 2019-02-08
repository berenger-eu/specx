///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

#include "Buffer/SpBufferDataView.hpp"
#include "Buffer/SpHeapBuffer.hpp"

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


class HeapBufferTest : public UTester< HeapBufferTest > {
    using Parent = UTester< HeapBufferTest >;

    void TestBasic(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        {
            SpHeapBuffer<TestClassWithCounter> heapBuffer(10);

            auto testBuffer = heapBuffer.getNewBuffer();

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

        UASSERTETRUE(TestClassWithCounter::GetNbCreated() == TestClassWithCounter::GetNbDeleted());
    }

    void TestSameBuffer(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        {
            SpHeapBuffer<TestClassWithCounter> heapBuffer(10);


            TestClassWithCounter* objectPtr = nullptr;

            int sequentialTaskFlow = 0;

            {
                auto testBuffer = heapBuffer.getNewBuffer();
                runtime.task(SpWrite(sequentialTaskFlow), SpWrite(testBuffer.getDataDep()),
                             [&objectPtr](int& /*sequentialTaskFlow*/, SpDataBuffer<TestClassWithCounter> testObject){
                    objectPtr = &(*testObject);
                });
            }

            for(int idx = 0 ; idx < 100 ; ++idx){
                auto testBuffer = heapBuffer.getNewBuffer();
                runtime.task(SpWrite(sequentialTaskFlow), SpWrite(testBuffer.getDataDep()),
                             [this, &objectPtr](int& /*sequentialTaskFlow*/, SpDataBuffer<TestClassWithCounter> testObject){
                    UASSERTETRUE(objectPtr == &(*testObject));
                });
            }

            runtime.waitAllTasks();

            runtime.stopAllThreads();

            UASSERTETRUE(TestClassWithCounter::GetNbCreated() != TestClassWithCounter::GetNbDeleted());
        }

        UASSERTETRUE(TestClassWithCounter::GetNbCreated() == TestClassWithCounter::GetNbDeleted());
    }

    void SetTests() {
        Parent::AddTest(&HeapBufferTest::TestBasic, "Basic test for vec type");
        Parent::AddTest(&HeapBufferTest::TestSameBuffer, "Basic test for vec type");
    }
};

// You must do this
TestClass(HeapBufferTest)


