#ifndef SPDEVICEDATA_HPP
#define SPDEVICEDATA_HPP

struct SpDeviceData {
	void* ptr;
	std::size_t size;
};

struct SpDeviceDataOp {
	void* (* const allocate)(void* hostPtr);
	void (* const copyFromHostToDevice)(void* devicePtr, void* hostPtr);
	void (* const copyFromDeviceToHost)(void* hostPtr, void* devicePtr);
	void (* const free)(void* devicePtr);
};

template <typename T>
void* SpDeviceDataAlloc(T* hostPtr);

template <typename T>
void SpDeviceDataCopyFromHostToDevice(void* devicePtr, T* hostPtr);

template <typename T>
void SpDeviceDataCopyFromDeviceToHost(void* devicePtr, T* hostPtr);

template <typename T>
void SpDeviceDataFree(void* devicePtr);

#endif
