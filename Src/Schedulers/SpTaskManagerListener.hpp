#ifndef SPTASKMANAGERLISTENER_HPP
#define SPTASKMANAGERLISTENER_HPP

class SpAbstractTask;

class SpTaskManagerListener {
public:
    virtual ~SpTaskManagerListener(){}
    virtual void thisTaskIsReady(SpAbstractTask*, const bool isNotCalledInAContextOfTaskCreation) = 0;
};


#endif

