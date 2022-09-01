///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"


class TestCopyMove : public UTester< TestCopyMove > {
    using Parent = UTester< TestCopyMove >;

    void TestBasic(){
        class NotMovableNotCopyableClass {
        public:
            NotMovableNotCopyableClass() = default;
            NotMovableNotCopyableClass(const NotMovableNotCopyableClass&) = delete;
            NotMovableNotCopyableClass(NotMovableNotCopyableClass&&) = delete;
            NotMovableNotCopyableClass& operator=(const NotMovableNotCopyableClass&) = delete;
            NotMovableNotCopyableClass& operator=(NotMovableNotCopyableClass&&) = delete;

            NotMovableNotCopyableClass* clone() const {
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
