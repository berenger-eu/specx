#include "SpUtils.hpp"

/** The id of the "current" calling thread */
thread_local long int GlobalThreadId = 0;
thread_local SpWorkerTypes::Type ThreadWorkerType;
thread_local long int ThreadDeviceId = -1;

/** Return the curren thread id */
long int SpUtils::GetThreadId(){
    return GlobalThreadId;
}

/** Set the thread id (should be call by the runtime */
void SpUtils::SetThreadId(const long int inThreadId){
    GlobalThreadId = inThreadId;
}

/** Set current thread Id */
void SpUtils::SetThreadType(const SpWorkerTypes::Type inType){
    ThreadWorkerType = inType;
}

/** Set current thread Id */
SpWorkerTypes::Type SpUtils::GetThreadType(){
    return ThreadWorkerType;
}

/** Return the curren thread device id */
long int SpUtils::GetDeviceId(){
    return ThreadDeviceId;
}

/** Set the device thread id (should be call by the runtime */
void SpUtils::SetDeviceId(const long int inThreadDeviceId){
    ThreadDeviceId = inThreadDeviceId;
}