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
#include "Utils/small_vector.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();
    SpRuntime runtime(NumThreads);

    for(int idx = 0 ; idx < 5 ; ++idx){
        SpBufferDataView<small_vector<int>> vectorBuffer;

        runtime.task(SpWrite(vectorBuffer.getDataDep()),
                     []([[maybe_unused]] SpDataBuffer<small_vector<int>> vector){
        });
        for(int idxSub = 0 ; idxSub < 3 ; ++idxSub){
            runtime.task(SpRead(vectorBuffer.getDataDep()),
                         []([[maybe_unused]] const SpDataBuffer<small_vector<int>> vector){
            });
        }
    }

    runtime.waitAllTasks();

    return 0;
}
