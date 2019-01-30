///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

#include "Buffer/SpBufferDataView.hpp"
#include "Buffer/SpHeapBuffer.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    SpHeapBuffer<std::vector<int>> heapBuffer;

    for(int idx = 0 ; idx < 5 ; ++idx){
        auto vectorBuffer = heapBuffer.getNewBuffer();

        runtime.task(SpWrite(vectorBuffer.getDataDep()),
                     [](SpDataBuffer<std::vector<int>> vector){
        });
        for(int idxSub = 0 ; idxSub < 3 ; ++idxSub){
            runtime.task(SpRead(vectorBuffer.getDataDep()),
                         [](const SpDataBuffer<std::vector<int>> vector){
            });
        }
    }

    runtime.waitAllTasks();

    return 0;
}
