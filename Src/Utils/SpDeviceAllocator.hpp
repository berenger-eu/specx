#ifndef SPDEVICEALLOCATOR_HPP
#define SPDEVICEALLOCATOR_HPP

#include <cstring>

class SpDeviceAllocator {
public:
	static void* allocateOnDevice(std::size_t size, std::size_t alignment) {
		if(alignment <= alignof(std::max_align_t)) {
			return std::malloc(size);
		} else {
			return std::aligned_alloc(size, alignment);
		}
	}
	
	static void copyHostToDevice(void* devicePtr, const void* hostPtr, std::size_t size) {
		std::memcpy(devicePtr, hostPtr, size);
	}
	
	static void copyDeviceToHost(void* hostPtr, const void* devicePtr, std::size_t size) {
		std::memcpy(hostPtr, devicePtr, size);
	}
	
	static void freeFromDevice(void* devicePtr) {
		std::free(devicePtr);
	}
};

#endif
