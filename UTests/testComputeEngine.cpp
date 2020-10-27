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
#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"
#include "Tasks/SpTask.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Compute/SpWorker.hpp"
#include "Utils/small_vector.hpp"

class ComputeEngineTest : public UTester< ComputeEngineTest > {
    using Parent = UTester< ComputeEngineTest >;

    void Test(){
        
        SpTaskGraph<SpSpeculativeModel::SP_MODEL_1> tg1, tg2;
        
        std::array<small_vector<std::unique_ptr<SpWorker>, 2>, 2> workerVectors;
        
        auto generateFunc =
        []() {
            return std::make_unique<SpWorker>(SpWorker::SpWorkerType::CPU_WORKER);
        };
        
        for(auto& workerVector : workerVectors) {
            workerVector.resize(2);
            std::generate(std::begin(workerVector), std::end(workerVector), generateFunc);
            for(auto& w : workerVector) {
                w->start();
            }
        }
        
        SpComputeEngine ce1(std::move(workerVectors[0])), ce2(std::move(workerVectors[1]));
        
        int a = 0, b = 0;
        
        std::promise<bool> tg1Promise;
        std::promise<bool> mainThreadPromise;
        
        tg1.task(SpWrite(a),
        [&](int& inA) {
            mainThreadPromise.set_value(true);
            tg1Promise.get_future().get();
            inA = 1;
        });
        
        tg1.task(SpRead(a), SpWrite(b),
        [](const int& inA, int& inB) {
           inB = inA;
        });
        
        std::array<std::promise<bool>, 4> promises;
        
        for(size_t i = 1; i < promises.size(); i++) {
            tg2.task(
            [&promises, i]() {
                promises[i].get_future().get();
                promises[i-1].set_value(true);
            }
            );
        }
        
        tg1.computeOn(ce1);
        
        mainThreadPromise.get_future().get();
        
        auto workers = ce1.detachWorkers(SpWorker::SpWorkerType::CPU_WORKER, 1, true);
        
        tg1Promise.set_value(true);
        
        UASSERTEEQUAL(static_cast<int>(workers.size()), 1);
        
        tg2.computeOn(ce2);
        
        promises[promises.size()-1].set_value(true);
        
        ce2.addWorkers(std::move(workers));
        
        tg2.waitAllTasks();
        
        ce2.sendWorkersTo(std::addressof(ce1), SpWorker::SpWorkerType::CPU_WORKER, 3, true);
        
        tg1.waitAllTasks();
        
        UASSERTEEQUAL(static_cast<int>(ce1.getCurrentNbOfWorkers()), 4);
        UASSERTEEQUAL(static_cast<int>(ce2.getCurrentNbOfWorkers()), 0);
        
        tg1.generateTrace("/tmp/taskgraph1.svg");
        tg2.generateTrace("/tmp/taskgraph2.svg");
        
    }

    void SetTests() {
        Parent::AddTest(&ComputeEngineTest::Test, "Compute engine test");
    }
};

// You must do this
TestClass(ComputeEngineTest)


