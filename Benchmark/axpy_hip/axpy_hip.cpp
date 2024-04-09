///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>
#include <iostream>

#include <clsimple.hpp>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"

#include "hip/hip_runtime.h"

 
#define SPECX_COMPILE_WITH_HIP





#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


//#define WIDTH     1024
//#define HEIGHT    1024

#define WIDTH     16
#define HEIGHT    16


#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1





template <class NumType>
struct Vector{
    std::vector<NumType> data;

    /////////////////////////////////////////////////////////////

    class DataDescr {
        std::size_t size;
    public:
        explicit DataDescr(const std::size_t inSize = 0) : size(inSize){}

        auto getSize() const{
            return size;
        }
    };

    using DataDescriptor = DataDescr;

    std::size_t memmovNeededSize() const{
        return sizeof(NumType)*data.size();
    }

    template <class DeviceMemmov>
    auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
        assert(size == sizeof(NumType)*data.size());
        NumType* doubleDevicePtr = reinterpret_cast<NumType*>(devicePtr);
        mover.copyHostToDevice(doubleDevicePtr, data.data(), sizeof(NumType)*data.size());
        return DataDescr(data.size());
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
        assert(size == sizeof(NumType)*data.size());
        NumType* doubleDevicePtr = reinterpret_cast<NumType*>(devicePtr);
        mover.copyDeviceToHost(data.data(), doubleDevicePtr, sizeof(NumType)*data.size());
    }

};



class MemmovClassExample{
    int data[10];
public:
    std::size_t memmovNeededSize() const{
        return 10*sizeof(int);
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10*sizeof(int));
        mover.copyHostToDevice(reinterpret_cast<int*>(devicePtr), &data[0], 10*sizeof(int));
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10*sizeof(int));
        mover.copyDeviceToHost(&data[0], reinterpret_cast<int*>(devicePtr), 10*sizeof(int));
    }

    struct DataDescr{
        DataDescr(){}
    };

    auto getDeviceDataDescription() const{
        return DataDescr();
    }
};


 
#ifdef SPECX_COMPILE_WITH_HIP
template <class NumType>
__global__ void cu_axpy(int n, NumType a, NumType *x, NumType *y, NumType *out)
{
    //int i = blockIdx.x*blockDim.x + threadIdx.x;
    int i = hipBlockDim_x*hipBlockIdx_x+hipThreadIdx_x;
    if (i < n)
        out[i] = a*x[i] + y[i];
}



__global__ void inc_var(int* ptr, int size){
    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
        //ptr[idx]++;
        ptr[idx]=9;
    }
}




__global__ void vector_add(double* __restrict__ a, const double* __restrict__ b, const double* __restrict__ c, int width, int height) 
  {
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
      int i = y * width + x;
      if ( i < (width * height)) {
        a[i] = b[i] + c[i];
      }
  }

#if 0
__kernel__ void vector_add(double* a, const double* b, const double* c, int width, int height) {
 
  int x = blockDimX * blockIdx.x + threadIdx.x;
  int y = blockDimY * blockIdy.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}
#endif


#endif


void check_solution(double* a_in,double* b_in,double* c_in,int nb)
{
    printf ("\n");
	int errors = 0;
  	for (int i = 0; i < nb; i++) {
	    if (a_in[i] != (b_in[i] + c_in[i])) { errors++; }
	}
  	if (errors!=0) { printf("FAILED: %d errors\n",errors); } else { printf ("WELL DONE PASSED! :-)\n"); }
}

void write_vector(char *ch,double* v,int nb)
{
	printf("%s\n",ch);
	for (int i = 0; i < nb; i++) { printf("%i",int(v[i])); }
	printf("\n");
}


 
void Test001()
{
    double* hostCpu_A;
    double* hostCpu_B;
    double* hostCpu_C;

    double* deviceGPU_A;
    double* deviceGPU_B;
    double* deviceGPU_C;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    std::cout << " System minor " << devProp.minor << std::endl;
    std::cout << " System major " << devProp.major << std::endl;
    std::cout << " agent prop name " << devProp.name << std::endl;
    
    hostCpu_A = (double*)malloc(NUM * sizeof(double));
    hostCpu_B = (double*)malloc(NUM * sizeof(double));
    hostCpu_C = (double*)malloc(NUM * sizeof(double));
    
    for (int i = 0; i < NUM; i++) {
        hostCpu_A[i] = 0;
        hostCpu_B[i] = 1;
        hostCpu_C[i] = 2;
    }
    
    HIP_ASSERT(hipMalloc((void**)&deviceGPU_A, NUM * sizeof(double)));
    HIP_ASSERT(hipMalloc((void**)&deviceGPU_B, NUM * sizeof(double)));
    HIP_ASSERT(hipMalloc((void**)&deviceGPU_C, NUM * sizeof(double)));
    
    HIP_ASSERT(hipMemcpy(deviceGPU_B, hostCpu_B, NUM*sizeof(double), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(deviceGPU_C, hostCpu_C, NUM*sizeof(double), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(vector_add, 
                    dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                    0, 0,
                    deviceGPU_A ,deviceGPU_B ,deviceGPU_C ,WIDTH ,HEIGHT);


    HIP_ASSERT(hipMemcpy(hostCpu_A, deviceGPU_A, NUM*sizeof(double), hipMemcpyDeviceToHost));

    // BEGIN::CTRL
    write_vector("Vector A",hostCpu_A,NUM);
    check_solution(hostCpu_A,hostCpu_B,hostCpu_C,NUM);
    // END::CTRL


    HIP_ASSERT(hipFree(deviceGPU_A));
    HIP_ASSERT(hipFree(deviceGPU_B));
    HIP_ASSERT(hipFree(deviceGPU_C));

    free(hostCpu_A);
    free(hostCpu_B);
    free(hostCpu_C);

}


void Test002(){
        SpHipUtils::PrintInfo();

        /*
        TeamOfCpuHipWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             int nbHipWorkers = SpHipUtils::GetNbDevices()) 
        */

        //SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(1,1,2));
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(SpUtils::DefaultNumThreads(),SpHipUtils::GetDefaultNbStreams(),SpHipUtils::GetNbDevices()));
        SpTaskGraph tg;
        std::vector<int> a(100,0);
        std::vector<int> b(100,0);

        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,
                      "should be stdvec");

        tg.computeOn(ce);

        tg.task(SpRead(a),
            SpHip([]([[maybe_unused]] SpDeviceDataView<const std::vector<int>> paramA) {
            })
        );

        tg.task(SpWrite(a),
            SpHip([](SpDeviceDataView<std::vector<int>> paramA) {
                inc_var<<<1,1,0,SpHipUtils::GetCurrentStream()>>>(paramA.array(),
                                                                   paramA.nbElements());

                std::this_thread::sleep_for(std::chrono::seconds(2));
            })
        );

        tg.task(SpWrite(b),
            SpHip([](SpDeviceDataView<std::vector<int>> paramB) {
                inc_var<<<1,1,0,SpHipUtils::GetCurrentStream()>>>(paramB.array(),
                    paramB.nbElements());
            })
        );

        if (false) {
            tg.task(SpRead(a), SpWrite(b),
                SpCpu([](const std::vector<int>& paramA, std::vector<int>& paramB) {
                    assert(paramA.size() == paramB.size());
                    for(int idx = 0 ; idx < int(paramA.size()) ; ++idx){
                        paramB[idx] = paramA[idx] + paramB[idx];
                    }
                })
            );

            tg.task(SpWrite(a),
                SpCpu([](std::vector<int>& paramA) {
                    for(auto& va : paramA){
                        va++;
                    }
                }),
                SpHip([](SpDeviceDataView<std::vector<int>> paramA) {
                    inc_var<<<1,1,0,SpHipUtils::GetCurrentStream()>>>(paramA.array(),
                                                                    paramA.nbElements());
                })
            );
        }    
              
        tg.task(SpWrite(a), SpWrite(b),
            SpCpu([](std::vector<int>& paramA, std::vector<int>& paramB) {
                if (false) {
                    for(auto& va : paramA){
                        va++;
                    }
                    for(auto& vb : paramB){
                        vb++;
                    }
                }
            })
        );

        tg.waitAllTasks();

        std::cout<<"va"<<"\n";
        for(auto& va : a){
            std::cout<<va;
        }
        std::cout<<"\n";

        std::cout<<"vb"<<"\n";
        for(auto& vb : b){
            std::cout<<vb;
        }
        std::cout<<"\n";
    }
 
 
void BenchmarkTest(int argc, char** argv){
//./axpy --sz=50 --th=256

    CLsimple args("Axpy", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int size = 100;
    args.addParameter<int>({"sz" ,"size"}, "Size", size, 1024);

    int nbthreads;
    args.addParameter<int>({"th"}, "nbthreads", nbthreads, 256);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return;
    }

    Vector<float> x;
    x.data.resize(size, 1);
    Vector<float> y;
    y.data.resize(size, 2);
    Vector<float> z;
    z.data.resize(size, 0);
    const float a = 2;


    /*
    std::cout<<"Vector x\n";
    for(int k=0;k<size;k++) {
        std::cout<<x.data[k];
    } 
    std::cout<<"\n";
    
    std::cout<<"Vector y\n";
    for(int k=0;k<size;k++) {
        std::cout<<y.data[k];
    } 
    std::cout<<"\n";
    */

#ifdef SPECX_COMPILE_WITH_HIP
    std::cout<<"In HIP Part..."<<"\n";
    SpHipUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers());
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
#endif
    SpTaskGraph tg;
    tg.computeOn(ce);
    tg.task(SpCommutativeWrite(z),SpRead(x),SpRead(y),
#ifdef SPECX_COMPILE_WITH_HIP
            SpHip([a, nbthreads](SpDeviceDataView<Vector<float>> paramZ,
                       const SpDeviceDataView<const Vector<float>> paramX,
                       const SpDeviceDataView<const Vector<float>> paramY) {
                const int size = paramZ.data().getSize();
                const int nbBlocks = (size + nbthreads-1)/nbthreads;
                cu_axpy<float>
                <<<nbBlocks, nbthreads,0,SpHipUtils::GetCurrentStream()>>>
                (size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());
            })
#else
            SpCpu([ta=a](Vector<float>& tz, const Vector<float>& tx, const Vector<float>& ty) {
                for(int idx = 0 ; idx < int(tz.data.size()) ; ++idx){
                    tz.data[idx] = ta*tx.data[idx]*ty.data[idx];
                }
            })
#endif
            );



    std::cout<<"In HIP Part 2..."<<"\n";
    
    tg.task(
        SpWrite(z),
            SpCpu(
                [&](Vector<float>& v)  
                {

                }
            )
    );

        
   std::cout<<"\n";
    for(int k=0;k<size;k++) {
        std::cout<<z.data[k];
    } 
    std::cout<<"\n";


/**/

#ifdef SPECX_COMPILE_WITH_HIP
    std::cout<<"In HIP Part 3..."<<"\n";
    tg.task(SpWrite(z),
    SpCpu([](Vector<float>&) {
    })
    );
#endif


    tg.waitAllTasks();

    //std::cout << "Generate trace ./axpy-simu.svg" << std::endl;
    //tg.generateTrace("./axpy-simu.svg", false);

}







int main(int argc, char** argv){

    std::cout<<"Test 001"<<"\n";
    Test001();


    std::cout<<"\n";

    std::cout<<"Test 001"<<"\n";
    Test002();


    std::cout<<"\n";
   
    std::cout<<"Test Benchmark"<<"\n";
    BenchmarkTest(argc, argv);

    std::cout<<"\n";
    std::cout<<"Finished..."<<"\n";

    return 0;
}
