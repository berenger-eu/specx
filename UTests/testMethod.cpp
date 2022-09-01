///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class MethodTest : public UTester< MethodTest > {
    using Parent = UTester< MethodTest >;

    void TestBasic(){
        const int NumThreads = SpUtils::DefaultNumThreads();
        SpRuntime runtime(NumThreads);

        const int initVal = 1;
        int writeVal = 0;

        class C_constint{
        public:
            C_constint() = default;
            C_constint(const C_constint&) = default;
            C_constint& operator=(const C_constint&) = default;
            C_constint(C_constint&&) = default;
            C_constint& operator=(C_constint&&) = default;

            void operator()(const int& /*initValParam*/) const{

            }
        };

        C_constint o1;
        {
            auto returnValue = runtime.task(SpRead(initVal), o1);
            returnValue.getValue();
        }

        class C_constint_intref{
        public:
            C_constint_intref() = default;
            C_constint_intref(const C_constint_intref&) = default;
            C_constint_intref& operator=(const C_constint_intref&) = default;
            C_constint_intref(C_constint_intref&&) = default;
            C_constint_intref& operator=(C_constint_intref&&) = default;

            bool operator()(const int& /*initValParam*/, int& /*writeValParam*/) const{
                return true;
            }
        };

        C_constint_intref o2;
        {
            auto returnValue = runtime.task(SpRead(initVal), SpWrite(writeVal), o2);
            returnValue.wait();
            UASSERTETRUE(returnValue.getValue() == true);
        }

        class C_intref{
        public:
            C_intref() = default;
            C_intref(const C_intref&) = default;
            C_intref& operator=(const C_intref&) = default;
            C_intref(C_intref&&) = default;
            C_intref& operator=(C_intref&&) = default;

            int operator()(const int& /*writeValParam*/) const{
                return 99;
            }
        };

        C_intref o3;
        {
            auto returnValue = runtime.task(SpRead(writeVal),o3);
            UASSERTETRUE(returnValue.getValue() == 99);
        }

        runtime.waitAllTasks();

        runtime.stopAllThreads();
    }

    void SetTests() {
        Parent::AddTest(&MethodTest::TestBasic, "Basic test for vec type");
    }
};

// You must do this
TestClass(MethodTest)


