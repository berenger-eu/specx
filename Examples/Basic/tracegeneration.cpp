///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    int initVal = 1;
    int writeVal = 0;
    int baseTime = 10000;

    runtime.task(SpRead(initVal),
                 [baseTime](const int&){
        usleep(1*baseTime);
    });

    runtime.task(SpRead(initVal), SpWrite(writeVal),
                 [baseTime](const int&, int&){
        usleep(3*baseTime);
    });

    runtime.task(SpRead(writeVal),
                 [baseTime](const int&){
        usleep(5*baseTime);
    });

    for(int idx = 0 ; idx < 5 ; ++idx){
        runtime.task(SpWrite(initVal), SpWrite(writeVal),
                     [baseTime](int&, int&){
            usleep(10*baseTime);
        });
        runtime.task(SpRead(writeVal),
                     [baseTime](const int&){
            usleep(1*baseTime);
        });
        runtime.task(SpWrite(initVal),
                     [baseTime](int&){
            usleep(10*baseTime);
        });

        usleep(baseTime/10);
    }

    runtime.waitAllTasks();

    runtime.stopAllThreads();

    runtime.generateTrace("/tmp/trace.svg");

    return 0;
}
