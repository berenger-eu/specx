///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

// @NBTESTS = 3

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    const int initVal = 1;
    int writeVal = 0;
    
#ifdef TEST1
    runtime.task(SpRead(initVal),
                 [](double& initValParam){
    });
#endif
#ifdef TEST2
    runtime.task(SpRead(initVal), SpWrite(writeVal),
                 [](const int& initValParam){
    });
#endif
#ifdef TEST3
    runtime.task(SpRead(writeVal),
                 [](const int* writeValParam){
    });
#endif

    runtime.waitAllTasks();

    runtime.stopAllThreads();
}


