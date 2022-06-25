#ifndef SPCUDAUTILS_HPP
#define SPCUDAUTILS_HPP

#include <iostream>
#include <cstring>
// TODO #include <cuda.h>

#define CUDA_ASSERT(X)\
    (X)
/*if ( cudaSuccess != (X) ){\
    printf("Error: fails, %s (%s line %d)\n", cudaGetErrorString(cuda_status), __FILE__, __LINE__ );\
    exit(1);\
}*/

class SpCudaUtils{
    static std::vector<bool> ConnectDevices(){
        const int nbDevices = GetNbCudaDevices();
        std::vector<bool> connected(nbDevices*nbDevices, false);
        for(int idxGpu1 = 0 ; idxGpu1 < nbDevices ; ++idxGpu1){
            UseDevice(idxGpu1);
            for(int idxGpu2 = 0 ; idxGpu2 < nbDevices ; ++idxGpu2){
                int is_able;
                cudaDeviceCanAccessPeer(&is_able, idxGpu1, idxGpu2);
                if(is_able){
                    cudaDeviceEnablePeerAccess(idxGpu2, 0);
                    connected[idxGpu1*nbDevices + idxGpu2] = true;
                    connected[idxGpu2*nbDevices + idxGpu1] = true;
                }
            }
        }
    }

    static std::vector<bool> ConnectedDevices;

public:
    static bool DevicesAreConnected(int idxGpu1, int idxGp2){
        return ConnectDevices[idxGpu1*GetNbCudaDevices() + idxGp2];
    }

    static int GetNbCudaDevices(){
        int nbDevices;
        nbDevices = 2;//CUDA_ASSERT(cudaGetDeviceCount(nbDevices));// TODO Ask CUDA
        return nbDevices;
    }

    static std::size_t GetTotalMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        // TODO CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte ));
        total_byte = 16 * 1024L * 1024L * 1024L;
        return total_byte;
    }

    static std::size_t GetFreeMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        // TODO CUDA_ASSERT(cudaMemGetInfo( &free_byte, &total_byte ));
        free_byte = 16 * 1024L * 1024L * 1024L;
        return free_byte;
    }

    static void UseDevice(const int deviceId){
        // TODO cudaSetDevice (deviceId);
    }

    static void SynchronizeDevice(){
        // TODO cudaDeviceSynchronize();
    }

    static void SynchronizeStream(){
        //cudaStreamSynchronize(stream1);
    }

    static int GetNbDevices(){
        int num;
        //cudaGetDeviceCount(&num);
        return num;
    }

    static int GetDefaultNbStreams(){
        return 4;
    }

    static int PrintDeviceName(const int gpuId){
        // TODO cudaDeviceProp prop;
        // cudaGetDeviceProperties(&prop, gpuId);
        // std::cout << "Device id: " << gpuId << std::endl;
        // std::cout << "Device name: " << prop.name << std::endl;
    }
};

#endif // SPCUDAUTILS_HPP
