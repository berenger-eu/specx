///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <future>
#include <algorithm>
#include <array>
#include <memory>

#include "UTester.hpp"
#include "Utils/SpConsumerThread.hpp"

class ConsumerThread : public UTester< ConsumerThread > {
    using Parent = UTester< ConsumerThread >;

    void Test(){
        int i = 0;
        {
            SpConsumerThread cs;

            cs.submitJobAndWait([&](){
                i++;
            });

            UASSERTEEQUAL(i, 1);

            cs.submitJobAndWait([&](){
                i++;
            });

            UASSERTEEQUAL(i, 2);

            cs.submitJob([&](){
                i++;
            });
            cs.submitJob([&](){
                i++;
            });
            cs.submitJobAndWait([&](){
                i++;
            });

            UASSERTEEQUAL(i, 5);

            cs.submitJob([&](){
                i++;
            });
        }

        UASSERTEEQUAL(i, 6);
    }

    void TestStopAndDrain(){
        SpConsumerThread cs;
        int count = 0;
        for(int idx = 0; idx < 100; ++idx) {
            cs.submitJob([&]() { ++count; });
        }
        cs.stop();
        cs.stop();
        UASSERTEEQUAL(count, 100);
        bool rejected = false;
        try {
            cs.submitJob([]() {});
        } catch(const std::runtime_error&) {
            rejected = true;
        }
        UASSERTETRUE(rejected);
        rejected = false;
        try {
            cs.submitJobAndWait([]() {});
        } catch(const std::runtime_error&) {
            rejected = true;
        }
        UASSERTETRUE(rejected);
    }

    void TestExceptionAndNestedSubmission(){
        SpConsumerThread cs;
        bool caught = false;
        try {
            cs.submitJobAndWait([]() { throw std::logic_error("job failed"); });
        } catch(const std::logic_error&) {
            caught = true;
        }
        UASSERTETRUE(caught);
        int value = 0;
        cs.submitJobAndWait([&]() {
            cs.submitJobAndWait([&]() { value = 42; });
        });
        UASSERTEEQUAL(value, 42);
    }

    void SetTests() {
        Parent::AddTest(&ConsumerThread::Test, "ConsumerThread");
        Parent::AddTest(&ConsumerThread::TestStopAndDrain, "Stop drains jobs and rejects further submissions");
        Parent::AddTest(&ConsumerThread::TestExceptionAndNestedSubmission, "Synchronous exceptions and nested submissions");
    }
};

// You must do this
TestClass(ConsumerThread)


