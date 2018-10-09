#include "SpUtils.hpp"

/** The id of the "current" calling thread */
thread_local long int GlobalThreadId = 0;

/** Return the curren thread id */
long int SpUtils::GetThreadId(){
    return GlobalThreadId;
}

/** Set the thread id (should be call by the runtime */
void SpUtils::SetThreadId(const long int inThreadId){
    GlobalThreadId = inThreadId;
}
