#ifndef SPGPUUNUSEDDATASTORE_HPP
#define SPGPUUNUSEDDATASTORE_HPP

#include <list>
#include <mutex>

class SpDataHandle;

class SpGpuUnusedDataStore {
private:
	std::list<SpDataHandle*> unusedDataHandles;
	std::mutex unusedDataHandlesMutex;
public:
	using iterator = std::list<SpDataHandle*>::iterator;
	
	iterator add(SpDataHandle* h) {
		std::unique_lock<std::mutex> lock(unusedDataHandlesMutex);
		unusedDataHandles.push_back(h);
		return unusedDataHandles.end();
	}
	
	void remove(iterator it) {
		std::unique_lock<std::mutex> lock(unusedDataHandlesMutex);
		unusedDataHandles.erase(it);
	}
};

#endif
