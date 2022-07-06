///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPPRIORITY_HPP
#define SPPRIORITY_HPP

/**
 * This class contains the priority of a task.
 * It should be used at task creation to inform the runtime
 * about the desired priority.
 */
class SpPriority{
    int priority;
public:
    explicit SpPriority(const int inPriority)
        : priority(inPriority){
    }
    
    int getPriority() const{
        return priority;
    }
};

#endif

