#ifndef SPHIPUTILS_HPP
#define SPHIPUTILS_HPP

#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>

#include "Config/SpConfig.hpp"

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#define HIP_ASSERT(X)\
{\
        hipError_t ___resHip = (X);\
        if ( hipSuccess != ___resHip ){\
            printf("Error: fails, %s (%s line %d)\n", hipGetErrorString(___resHip), __FILE__, __LINE__ );\
            exit(1);\
    }\
}

class SpHipUtils{
    static std::vector<bool> ConnectDevices(){
        const int nbDevices = GetNbDevices();
        std::vector<bool> connected(nbDevices*nbDevices, false);
        for(int idxHip1 = 0 ; idxHip1 < nbDevices ; ++idxHip1){
            UseDevice(idxHip1);
            for(int idxHip2 = 0 ; idxHip2 < nbDevices ; ++idxHip2){
                int is_able;
                HIP_ASSERT(hipDeviceCanAccessPeer(&is_able, idxHip1, idxHip2));
                if(is_able){
                    HIP_ASSERT(hipDeviceEnablePeerAccess(idxHip2, 0));
                    connected[idxHip1*nbDevices + idxHip2] = true;
                    connected[idxHip2*nbDevices + idxHip1] = true;
                }
            }
        }
        return connected;
    }

    static std::vector<bool> ConnectedDevices;

public:
    static bool DevicesAreConnected(int idxHip1, int idxHip2){
        return ConnectedDevices[idxHip1*GetNbDevices() + idxHip2];
    }

    static std::size_t GetTotalMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        HIP_ASSERT(hipMemGetInfo( &free_byte, &total_byte ));
        return total_byte;
    }

    static std::size_t GetFreeMemOnDevice(){
        size_t free_byte ;
        size_t total_byte ;
        HIP_ASSERT(hipMemGetInfo( &free_byte, &total_byte ));
        return free_byte;
    }

    static void UseDevice(const int deviceId){
        if(deviceId >= GetNbDevices()){
            std::cerr << "[SPECX] Ask to use gpu " << deviceId
                      << " but there are only " << GetNbDevices() << " gpus" << std::endl;
        }
        HIP_ASSERT(hipSetDevice (deviceId));
    }

    static void SynchronizeDevice(){
        HIP_ASSERT(hipDeviceSynchronize());
    }

    static void SynchronizeStream(hipStream_t& stream){
        HIP_ASSERT(hipStreamSynchronize(stream));
    }

    static int GetNbDevices(){
        int num;
        HIP_ASSERT(hipGetDeviceCount(&num));
        if(getenv("SPECX_NB_HIP_GPUS")){
            std::istringstream iss(getenv("SPECX_NB_HIP_GPUS"),std::istringstream::in);
            int nbGpus = -1;
            iss >> nbGpus;
            if( /*iss.tellg()*/ iss.eof() ) return std::min(nbGpus, num);
        }
        return num;
    }

    static int GetDefaultNbStreams(){
        if(getenv("SPECX_NB_HIP_STREAMS")){
            std::istringstream iss(getenv("SPECX_NB_HIP_STREAMS"),std::istringstream::in);
            int nbStreams = -1;
            iss >> nbStreams;
            if( /*iss.tellg()*/ iss.eof() ) return nbStreams;
        }
        return 4;
    }

    static void PrintDeviceName(const int hipId){
        hipDeviceProp_t prop;
        HIP_ASSERT(hipGetDeviceProperties(&prop, hipId));
        std::cout << "[SPECX] - Device id: " << hipId << std::endl;
        std::cout << "[SPECX]   Device name: " << prop.name << std::endl;
    }

    static std::string GetDeviceName(const int hipId){
        hipDeviceProp_t prop;
        HIP_ASSERT(hipGetDeviceProperties(&prop, hipId));
        return prop.name;
    }

    static void PrintInfo(){
        const int nbGpus = GetNbDevices();
        std::cout << "[SPECX] There are " << nbGpus << " gpus" << std::endl;
        for(int idxGpu = 0 ; idxGpu < nbGpus ; ++idxGpu){
            PrintDeviceName(idxGpu);
        }
    }

    static hipStream_t& GetCurrentStream();
    static bool CurrentWorkerIsHip();
    static int CurrentHipId();
    static void SyncCurrentStream();
};

#endif // SPHIPUTILS_HPP
