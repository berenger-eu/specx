#ifndef SPSCHEDULERINFORMER_HPP
#define SPSCHEDULERINFORMER_HPP

class SpAbstractTask;

class SpAbstractToKnowReady{
public:
    virtual ~SpAbstractToKnowReady(){}
    virtual void thisTaskIsReady(SpAbstractTask*, const bool isNotCalledInAContextOfTaskCreation) = 0;
};


#endif

