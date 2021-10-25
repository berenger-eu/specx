#ifndef SPHARDWARE_HPP
#define SPHARDWARE_HPP

#include <mutex>
#include <list>
#include "Utils/SpGpuUnusedDataStore.hpp"

class SpDataHandle;

namespace SpHardware {
	constexpr std::size_t nbGpus = 1;
	inline std::mutex gpuMutexes[nbGpus];
	inline SpGpuUnusedDataStore SpGpuUnusedDataStores[nbGpus];
}

#endif
