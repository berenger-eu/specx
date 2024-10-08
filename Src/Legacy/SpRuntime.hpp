///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPRUNTIME_HPP
#define SPRUNTIME_HPP

#include "Speculation/SpSpeculativeModel.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Compute/SpComputeEngine.hpp"
#include "Utils/SpUtils.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"

//! The runtime is the main component of specx.
template <SpSpeculativeModel SpecModel = SpSpeculativeModel::SP_MODEL_1>
class SpRuntime {
 
//=====----------------------------=====
//         Private API - members
//=====----------------------------=====
private:
    SpTaskGraph<SpecModel> tg;
    SpComputeEngine ce;

//=====--------------------=====
//          Public API
//=====--------------------=====
public:
    ///////////////////////////////////////////////////////////////////////////
    /// Constructor
    ///////////////////////////////////////////////////////////////////////////

    explicit SpRuntime(const int inNumThreads) :
            tg(), ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(inNumThreads)) {
        tg.computeOn(ce);
    }

#ifdef SPECX_COMPILE_WITH_CUDA
    SpRuntime() :
            tg(), ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers()) {
        tg.computeOn(ce);
    }
#else
    explicit SpRuntime() :
            tg(), ce(SpWorkerTeamBuilder::TeamOfCpuWorkers()) {
        tg.computeOn(ce);
    }
#endif
    ///////////////////////////////////////////////////////////////////////////
    /// Destructor
    ///////////////////////////////////////////////////////////////////////////
    ~SpRuntime() {
        tg.waitAllTasks();
        ce.stopIfNotAlreadyStopped();
    }
    
    // No copy and no move
    SpRuntime(const SpRuntime&) = delete;
    SpRuntime(SpRuntime&&) = delete;
    SpRuntime& operator=(const SpRuntime&) = delete;
    SpRuntime& operator=(SpRuntime&&) = delete;

    ///////////////////////////////////////////////////////////////////////////
    /// Task creation method
    ///////////////////////////////////////////////////////////////////////////

    template <class... ParamsTy>
    auto task(ParamsTy&&...params) {
        return tg.task(std::forward<ParamsTy>(params)...);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Getters/actions
    ///////////////////////////////////////////////////////////////////////////
    
    void setSpeculationTest(std::function<bool(int,const SpProbability&)> inFormula){
        tg.setSpeculationTest(std::move(inFormula));
    }
  
    void waitAllTasks(){
        tg.waitAllTasks();
    }
    
    void waitRemain(const long int windowSize){
        tg.waitRemain(windowSize);
    }

    void stopAllThreads(){
        ce.stopIfNotAlreadyStopped();
    }
    
    int getNbThreads() const {
        return static_cast<int>(ce.getCurrentNbOfWorkers());
    }

    int getNbCpuWorkers() const {
        return static_cast<int>(ce.getNbCpuWorkers());
    }
           
    template <class ClassFunc>
    void execOnWorkers(ClassFunc&& func) {
        ce.execOnWorkers(std::forward<ClassFunc>(func));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Output
    ///////////////////////////////////////////////////////////////////////////

    void generateDot(const std::string& outputFilename, bool printAccesses=false) const {
        tg.generateDot(outputFilename, printAccesses);
    }

    void generateTrace(const std::string& outputFilename, const bool showDependences = true) const {
        tg.generateTrace(outputFilename, showDependences);
    }
};
#endif
