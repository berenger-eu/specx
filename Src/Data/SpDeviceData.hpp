#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

#include <type_traits>
#include "Utils/SpDeviceAllocator.hpp"
#include "Utils/SpGpuUnusedDataStore.hpp"

struct SpDeviceData {
	void* ptr;
	long int useCount;
	long int cudaEvent;
	bool isInUnusedList;
	SpGpuUnusedDataStore::iterator it;
	
	SpDeviceData() : ptr(nullptr), useCount(0), cudaEvent(0), isInUnusedList(false), it() {}
};

struct SpDeviceDataOp {
	void* (* allocate)(void* hostPtr);
	void  (* copyFromHostToDevice)(void* devicePtr, void* hostPtr);
	void  (* copyFromDeviceToHost)(void* hostPtr, void* devicePtr);
	void  (* free)(void* devicePtr);
};

inline void* SpDeviceDataAlloc([[maybe_unused]] void* hostPtr) { return nullptr; }
inline void SpDeviceDataCopyFromHostToDevice([[maybe_unused]] void* devicePtr,[[maybe_unused]] void* hostPtr) {}
inline void SpDeviceDataCopyFromDeviceToHost([[maybe_unused]] void* hostPtr, [[maybe_unused]] void* devicePtr) {}

template<typename T>
void SpDeviceDataFree([[maybe_unused]] void* devicePtr) {}

template <typename T>
using SpDeviceDataTrivialCopyTest = std::conjunction<std::is_trivially_copy_constructible<T>, std::is_trivially_copy_assignable<T>>;

template <typename T>
void* SpDeviceDataAllocInternal([[maybe_unused]] T* hostPtr) {
	if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
		return SpDeviceAllocator::allocateOnDevice(sizeof(T), alignof(T));
	} else {
		return SpDeviceDataAlloc(hostPtr);
	}
}

template <typename T>
void SpDeviceDataCopyFromHostToDeviceInternal(void* devicePtr, T* hostPtr) {
	if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
		return SpDeviceAllocator::copyHostToDevice(devicePtr, hostPtr,  sizeof(T));
	} else {
		SpDeviceDataCopyFromHostToDevice(devicePtr, hostPtr);
	}
}

template <typename T>
void SpDeviceDataCopyFromDeviceToHostInternal(T* hostPtr, void* devicePtr) {
	if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
		return SpDeviceAllocator::copyDeviceToHost(hostPtr, devicePtr,  sizeof(T));
	} else {
		SpDeviceDataCopyFromDeviceToHost(hostPtr, devicePtr);
	}
}

template <typename T>
void SpDeviceDataFreeInternal(void* devicePtr) {
	if constexpr(SpDeviceDataTrivialCopyTest<T>::value) {
		return SpDeviceAllocator::freeFromDevice(devicePtr);
	} else {
		SpDeviceDataFree<T>(devicePtr);
	}
}

#endif
