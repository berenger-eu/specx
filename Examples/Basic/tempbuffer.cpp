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

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    for(int idx = 0 ; idx < 5 ; ++idx){
        SpBufferDataView<std::vector<int>> vectorBuffer;

        runtime.task(SpWrite(vectorBuffer.getDataDep()),
                     []([[maybe_unused]] SpDataBuffer<std::vector<int>> vector){
        });
        for(int idxSub = 0 ; idxSub < 3 ; ++idxSub){
            runtime.task(SpRead(vectorBuffer.getDataDep()),
                         []([[maybe_unused]] const SpDataBuffer<std::vector<int>> vector){
            });
        }
    }

    runtime.waitAllTasks();

    return 0;
}
