///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under MIT Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"


class TestCopyMove : public UTester< TestCopyMove > {
    using Parent = UTester< TestCopyMove >;

    void TestBasic(){
        class NotMovableNotCopyableClass {
        public:
            NotMovableNotCopyableClass() = default;
            NotMovableNotCopyableClass(const NotMovableNotCopyableClass&) = delete;
            NotMovableNotCopyableClass(const NotMovableNotCopyableClass&&) = delete;
            NotMovableNotCopyableClass& operator=(const NotMovableNotCopyableClass&) = delete;
            NotMovableNotCopyableClass& operator=(const NotMovableNotCopyableClass&&) = delete;

            NotMovableNotCopyableClass* clone() const{
                return nullptr;
            }
        };


        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        NotMovableNotCopyableClass testclass;
        {
            auto descriptor = runtime.task(SpRead(testclass),
                         [this, &testclass](const NotMovableNotCopyableClass& testclassParam){
                UASSERTETRUE(&testclassParam == &testclass);
            });

            descriptor.wait();
        }

        {
            auto descriptor = runtime.task(SpWrite(testclass),
                         [this, &testclass](NotMovableNotCopyableClass& testclassParam){
                UASSERTETRUE(&testclassParam == &testclass);
            });

            descriptor.wait();
        }
    }

    void SetTests() {
        Parent::AddTest(&TestCopyMove::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(TestCopyMove)
