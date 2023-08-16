#ifndef SPABSTRACTSCHEDULER_HPP
#define SPABSTRACTSCHEDULER_HPP

#include "Utils/small_vector.hpp"
#include "Compute/SpWorker.hpp"

class SpAbstractTask;

class SpAbstractScheduler{
public:
    virtual ~SpAbstractScheduler(){}

    virtual int getNbReadyTasksForWorkerType(const SpWorkerTypes::Type wt) const = 0;

    virtual int push(SpAbstractTask* newTask) = 0;

    virtual int pushTasks(small_vector_base<SpAbstractTask*>& tasks)  = 0;

    virtual SpAbstractTask* popForWorkerType(const SpWorkerTypes::Type wt) = 0;
};


#endif // SPABSTRACTSCHEDULER_HPP
