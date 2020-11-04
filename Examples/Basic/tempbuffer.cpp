///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

#include "Utils/SpBufferDataView.hpp"
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
