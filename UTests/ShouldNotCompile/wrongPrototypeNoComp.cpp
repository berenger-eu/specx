///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

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


