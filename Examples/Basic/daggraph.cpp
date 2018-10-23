///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    int initVal = 1;
    int writeVal = 0;

    runtime.task(SpRead(initVal),
                 [&initVal](const int& /*initValParam*/){
    });

    runtime.task(SpRead(initVal), SpWrite(writeVal),
                 [](const int& /*initValParam*/, int& /*writeValParam*/){
    });

    runtime.task(SpRead(writeVal),
                 [&initVal](const int& /*writeValParam*/){
    });

    for(int idx = 0 ; idx < 5 ; ++idx){
        runtime.task(SpWrite(initVal), SpWrite(writeVal),
                     [](int& /*initValParam*/, int& /*writeValParam*/){
        });
        runtime.task(SpRead(writeVal),
                     [&initVal](const int& /*writeValParam*/){
        });
    }

    runtime.waitAllTasks();

    runtime.stopAllThreads();

    runtime.generateDot("/tmp/graph.dot");

    return 0;
}
