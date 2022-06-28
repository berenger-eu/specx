///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
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

    void SetTests() {
        Parent::AddTest(&ConsumerThread::Test, "ConsumerThread");
    }
};

// You must do this
TestClass(ConsumerThread)


