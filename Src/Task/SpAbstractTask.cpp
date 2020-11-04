#include "SpAbstractTask.hpp"

// To store the task ids
// It is atomic but it should not be needed since the task
// must be created by the main thread only.
std::atomic<long int> SpAbstractTask::TaskIdsCounter(0);
