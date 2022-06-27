#ifndef SPCUDAUTILS_HPP
#define SPCUDAUTILS_HPP

#include <iostream>
#include <cstring>
#include <vector>


#include <cuda_runtime.h>
#include <cuda.h>

#define CUDA_ASSERT(X)\
{\
    cudaError_t ___resCuda = (X);\
    if ( cudaSuccess != ___resCuda ){\
        printf("Error: fails, %s (%s line %d)\n", cudaGetErrorString(___resCuda), __FILE__, __LINE__ );\
        exit(1);\
    }\
}

class SpCudaUtils{
    static std::vector<bool> ConnectDevices(){
        const int nbDevices = GetNbCudaDevices();
        std::vector<bool> connected(nbDevices*nbDevices, false);
        for(int idxGpu1 = 0 ; idxGpu1 < nbDevices ; ++idxGpu1){
            UseDevice(idxGpu1);
            for(int idxGpu2 = 0 ; idxGpu2 < nbDevices ; ++idxGpu2){
                int is_able;
                CUDA_ASSERT(cudaDeviceCanAccessPeer(&is_able, idxGpu1, idxGpu2));
                if(is_able){
                    CUDA_ASSERT(cudaDeviceEnablePeerAccess(idxGpu2, 0));
                    connected[idxGpu1*nbDevices + idxGpu2] = true;
                    connected[idxGpu2*nbDevices + idxGpu1] = true;
                }
            }
        }
        return connected;
    }

    static std::vector<bool> ConnectedDevices;

public:
    static bool DevicesAreConnected(int idxGpu1, int idxGp2){
        return ConnectedDevices[idxGpu1*GetNbCudaDevices() + idxGp2];
    }

    static int GetNbCudaDevices(){
        int nbDevices;
        CUDA_ASSERT(cudaGetDeviceCount(&nbDevices));
        return nbDevices;
    }

    static std::size_t GetTotalMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte ));
        return total_byte;
    }

    static std::size_t GetFreeMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte ));
        return free_byte;
    }

    static void UseDevice(const int deviceId){
        CUDA_ASSERT(cudaSetDevice (deviceId));
    }

    static void SynchronizeDevice(){
        CUDA_ASSERT(cudaDeviceSynchronize());
    }

    static void SynchronizeStream(cudaStream_t& stream){
        CUDA_ASSERT(cudaStreamSynchronize(stream));
    }

    static int GetNbDevices(){
        int num;
        CUDA_ASSERT(cudaGetDeviceCount(&num));
        return num;
    }

    static int GetDefaultNbStreams(){
        return 4;
    }

    static void PrintDeviceName(const int gpuId){
        cudaDeviceProp prop;
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, gpuId));
        std::cout << "Device id: " << gpuId << std::endl;
        std::cout << "Device name: " << prop.name << std::endl;
    }

    static cudaStream_t& GetCurrentStream();
    static bool CurrentWorkerIsGpu();
    static int CurrentGpuId();
    static void SyncCurrentStream();
};

#endif // SPCUDAUTILS_HPP
