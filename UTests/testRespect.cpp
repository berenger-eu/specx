///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <future>

#include "UTester.hpp"

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

class TestRespect : public UTester< TestRespect > {
    using Parent = UTester< TestRespect >;

    void TestBasic(){
        SpRuntime runtime(2);

        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            const int initVal = 1;

            runtime.task(SpRead(initVal),
                         [&](const int& /*initValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });
            
            promise1.get_future().wait();
            
            runtime.task(SpRead(initVal),
                         [&](const int& /*initValParam*/){
                promise2.set_value(0);
            });        
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;
        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            const int initVal = 1;
            int writeVal = 0;

            runtime.task(SpRead(initVal), SpWrite(writeVal),
                         [&](const int& /*initValParam*/, int& /*writeValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });
            
            promise1.get_future().wait();
            
            runtime.task(SpRead(initVal),
                         [&](const int& /*initValParam*/){
                promise2.set_value(0);
            });        
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;
        {
            std::promise<int> promise2;

            int writeVal = 0;

            runtime.task(SpWrite(writeVal),
                         [&](int& /*writeValParam*/){
                promise2.get_future().wait();
            });
            
            auto descriptor = runtime.task(SpRead(writeVal),
                         [&](const int& /*writeValParam*/){
            });
            UASSERTETRUE(descriptor.isReady() == false);
            
            promise2.set_value(0);       
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;        
        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            const int initVal[10] = {0};

            runtime.task(SpReadArray(initVal, SpArrayView(10)),
                         [&](const SpArrayAccessor<const int>& /*initValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });
            
            promise1.get_future().wait();
            
            runtime.task(SpReadArray(initVal, SpArrayView(10)),
                         [&](const SpArrayAccessor<const int>& /*initValParam*/){
                promise2.set_value(0);
            });
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;
        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            const int initVal[10] = {0};

            runtime.task(SpReadArray(initVal, SpArrayView(10).removeItems(5,9)),
                         [&](const SpArrayAccessor<const int>& /*initValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });
            
            promise1.get_future().wait();
            
            runtime.task(SpRead(initVal[0]),
                         [&](const int& /*initValParam*/){
                promise2.set_value(0);
            });        
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;
        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            int initVal[10] = {0};

            runtime.task(SpReadArray(initVal, SpArrayView(10).removeItems(1)),
                         [&](const SpArrayAccessor<const int>& /*initValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });
            
            promise1.get_future().wait();
            
            runtime.task(SpWrite(initVal[1]),
                         [&](int& /*initValParam*/){
                promise2.set_value(0);
            });        
            
            runtime.waitAllTasks();
        }
        std::cout << "Next" << std::endl;
        {
            std::promise<int> promise1;
            std::promise<int> promise2;

            int initVal[10] = {0};

            runtime.task(SpReadArray(initVal, SpArrayView(10).removeItems(0)),
                         [&](const SpArrayAccessor<const int>& /*initValParam*/){
                promise1.set_value(0);
                promise2.get_future().wait();
            });

            promise1.get_future().wait();

            runtime.task(SpWrite(initVal[0]),
                         [&](int& /*initValParam*/){
                promise2.set_value(0);
            });

            runtime.waitAllTasks();
        }

        runtime.stopAllThreads();
    }


    void TestMaxParallel(){
        const int dumbVal = 0;

        SpRuntime runtime(10);

        std::promise<long int> promises_A[10];
        for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
            runtime.task(SpRead(dumbVal),
                         [&,idxThread](const int& /*dumbValParam*/){
                promises_A[idxThread].set_value(idxThread);
                const long int res = promises_A[(idxThread+1)%10].get_future().get();
                UASSERTETRUE(res == (idxThread+1)%10);
            });
        }

        std::promise<long int> promises_B[10];
        for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
            runtime.task(SpRead(dumbVal),
                         [&,idxThread](const int& /*dumbValParam*/){
                promises_B[idxThread].set_value(idxThread);
                const long int res = promises_B[(idxThread+1)%10].get_future().get();
                UASSERTETRUE(res == (idxThread+1)%10);
            });
        }


        std::promise<long int> promises_C[10];
        for(int idxThread = 0 ; idxThread < runtime.getNbThreads() ; ++idxThread){
            runtime.task(SpRead(dumbVal),
                         [&,idxThread](const int& /*dumbValParam*/){
                promises_C[idxThread].set_value(idxThread);
                const long int res = promises_C[(idxThread+1)%10].get_future().get();
                UASSERTETRUE(res == (idxThread+1)%10);
            });
        }


        std::promise<long int> promises_D[5];
        for(int idxThread = 0 ; idxThread < runtime.getNbThreads()/2 ; ++idxThread){
            runtime.task(SpRead(dumbVal),
                         [&,idxThread](const int& /*dumbValParam*/){
                promises_D[idxThread].set_value(idxThread);
                const long int res = promises_D[(idxThread+1)%5].get_future().get();
                UASSERTETRUE(res == (idxThread+1)%5);
            });
        }

        runtime.waitAllTasks();
    }

    void SetTests() {
        Parent::AddTest(&TestRespect::TestBasic, "Basic tests");
        Parent::AddTest(&TestRespect::TestBasic, "Basic test max parallel");
    }
};

// You must do this
TestClass(TestRespect)


