#include "SpDebug.hpp"
#include "SpUtils.hpp"

SpDebug SpDebug::Controller;

long int SpDebug::getThreadId() const{
    return SpUtils::GetThreadId();
}
