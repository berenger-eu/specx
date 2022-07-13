#ifndef SPCUDAUTILS_HPP
#define SPCUDAUTILS_HPP

#include <iostream>
#include <cstring>
#include <vector>

#ifndef SPECX_COMPILE_WITH_CUDA
#error SPECX_COMPILE_WITH_CUDA must be defined
#endif

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
        const int nbDevices = GetNbDevices();
        std::vector<bool> connected(nbDevices*nbDevices, false);
        for(int idxCuda1 = 0 ; idxCuda1 < nbDevices ; ++idxCuda1){
            UseDevice(idxCuda1);
            for(int idxCuda2 = 0 ; idxCuda2 < nbDevices ; ++idxCuda2){
                int is_able;
                CUDA_ASSERT(cudaDeviceCanAccessPeer(&is_able, idxCuda1, idxCuda2));
                if(is_able){
                    CUDA_ASSERT(cudaDeviceEnablePeerAccess(idxCuda2, 0));
                    connected[idxCuda1*nbDevices + idxCuda2] = true;
                    connected[idxCuda2*nbDevices + idxCuda1] = true;
                }
            }
        }
        return connected;
    }

    static std::vector<bool> ConnectedDevices;

public:
    static bool DevicesAreConnected(int idxCuda1, int idxCuda2){
        return ConnectedDevices[idxCuda1*GetNbDevices() + idxCuda2];
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
        if(deviceId >= GetNbDevices()){
            std::cerr << "[SPECX] Ask to use gpu " << deviceId
                      << " but there are only " << GetNbDevices() << " gpus" << std::endl;
        }
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

    static void PrintDeviceName(const int cudaId){
        cudaDeviceProp prop;
        CUDA_ASSERT(cudaGetDeviceProperties(&prop, cudaId));
        std::cout << "[SPECX] - Device id: " << cudaId << std::endl;
        std::cout << "[SPECX]   Device name: " << prop.name << std::endl;
    }

    static void PrintInfo(){
        const int nbGpus = GetNbDevices();
        std::cout << "[SPECX] There are " << nbGpus << " gpus" << std::endl;
        for(int idxGpu = 0 ; idxGpu < nbGpus ; ++idxGpu){
            PrintDeviceName(idxGpu);
        }
    }

    static cudaStream_t& GetCurrentStream();
    static bool CurrentWorkerIsCuda();
    static int CurrentCudaId();
    static void SyncCurrentStream();
};

#endif // SPCUDAUTILS_HPP
