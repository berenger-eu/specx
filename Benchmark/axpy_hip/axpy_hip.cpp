#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunknown-attributes"
#pragma clang diagnostic ignored "-Wunknown-attributes"



///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>
#include <iostream>

//#include <clsimple.hpp>



#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"
#include "Utils/SpTimer.hpp"

#include "hip/hip_runtime.h"

 

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

__global__ void my_AMD(int* ptr, int size){
   int i = hipBlockDim_x*hipBlockIdx_x+hipThreadIdx_x;
   if (i < size)
        ptr[i]=9;
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
    /*
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
    */

}


void Test002A() {

        hipDeviceProp_t devProp;
        hipGetDeviceProperties(&devProp, 0);

        std::cout << "Device name " << devProp.name << std::endl;


        SpHipUtils::PrintInfo();

        std::cout<<"nbCpuWorkers="<<SpUtils::DefaultNumThreads()<<"\n";
        std::cout<<"GetDefaultNbStreams="<<SpHipUtils::GetDefaultNbStreams()<<"\n";       
        std::cout<<"GetNbDevices="<<SpHipUtils::GetNbDevices()<<"\n";

        /*
        TeamOfCpuHipWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             int nbHipWorkers = SpHipUtils::GetNbDevices()) 
        */

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(SpUtils::DefaultNumThreads(),1,SpHipUtils::GetNbDevices()));
        SpTaskGraph tg;
        int nx=10;
        std::vector<int> a(nx,0);

        const int block_size=SpUtils::DefaultNumThreads();
        int grid_size=(nx+block_size-1)/block_size;


        std::cout<<"va"<<"\n";
        for(auto& va : a){
            std::cout<<va;
        }
        std::cout<<"\n";


        //SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(grid_size,block_size,3));
        //SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(SpUtils::DefaultNumThreads(),SpHipUtils::GetDefaultNbStreams(),SpHipUtils::GetNbDevices()));

        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,"should be stdvec");

        tg.computeOn(ce);


        tg.task(SpRead(a),
            SpHip([&]([[maybe_unused]] SpDeviceDataView<const std::vector<int>> paramA) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                std::cout<<"NbElements="<<paramA.nbElements()<<"\n";
                //std::cout<<"Array="<<paramA.array()<<"\n";
            })
        );


        tg.task(SpWrite(a),
            SpHip([&](SpDeviceDataView<std::vector<int>> paramA) {
                //inc_var<<<1,1,0,SpHipUtils::GetCurrentStream()>>>(paramA.array(),paramA.nbElements());
                std::cout<<"NbElements="<<paramA.nbElements()<<"\n";
                std::cout<<"Array="<<paramA.array()<<"\n";
                hipLaunchKernelGGL(my_AMD,grid_size,block_size,0,SpHipUtils::GetCurrentStream(),paramA.array(),paramA.nbElements());
            })            
        );


        tg.task(SpWrite(a),
            SpCpu([](std::vector<int>& paramA) {
            })
        );

        tg.waitAllTasks();


        std::cout<<"va"<<"\n";
        for(auto& va : a){
            std::cout<<va;
        }
        std::cout<<"\n";
};



 /*
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

    std::cout<<"In HIP Part..."<<"\n";
    SpHipUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers());

    static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,
                      "should be stdvec");

    SpTaskGraph tg;
    tg.computeOn(ce);
    tg.task(SpCommutativeWrite(z),SpRead(x),SpRead(y),

            SpHip([a, nbthreads](SpDeviceDataView<Vector<float>> paramZ,
                       const SpDeviceDataView<const Vector<float>> paramX,
                       const SpDeviceDataView<const Vector<float>> paramY) {

                std::cout<<"HELLO\n";
                std::cout<<"NbElements="<<paramZ.getRawSize()<<"\n";
                std::cout<<"NbElements="<<paramX.getRawSize()<<"\n";

                //const int size = paramZ.data().getSize();
                //std::cout<<"NbElements="<<size<<"\n";
                const int size = 100;
                
                const int nbBlocks = (size + nbthreads-1)/nbthreads;

                //cu_axpy<float>
                //<<<nbBlocks, nbthreads,0,SpHipUtils::GetCurrentStream()>>>
                //(size, a, (float*)paramX.getRawPtr(), (float*)paramY.getRawPtr(), (float*)paramZ.getRawPtr());

                hipLaunchKernelGGL(cu_axpy<float>,nbBlocks, nbthreads,0,SpHipUtils::GetCurrentStream(),
                 size, a,
                 (float*)paramX.getRawPtr(), 
                 (float*)paramY.getRawPtr(), 
                 (float*)paramZ.getRawPtr()
                );

            })

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
*/

/*
class MemmovClassExample{
public:
    std::size_t memmovNeededSize() const{
        return 10;
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10);
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == 10);
    }

    struct View{
        View(){}
        View(void* devicePtr, std::size_t size){}
    };
    using DeviceDataType = View;
};
*/


/*
void TestMemMove()
{
    
        SpHipUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers(1,1,2));
        SpTaskGraph tg;
        tg.computeOn(ce);

        static_assert(SpDeviceDataView<MemmovClassExample>::MoveType == SpDeviceDataUtils::DeviceMovableType::MEMMOV,
                      "should be memmov");

        MemmovClassExample obj;

        
        tg.task(SpRead(obj),
            SpHip([]([[maybe_unused]] SpDeviceDataView<const MemmovClassExample> objv) {
            })
        );
        

        tg.task(SpWrite(obj),
            SpHip([](SpDeviceDataView<MemmovClassExample> objv) {
            })
        );



        tg.waitAllTasks(); 
    }
*/



int main(int argc, char** argv){


    //TestMemMove();

    //BenchmarkTest(argc, argv);

    Test002A();

/*
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
*/



    return 0;
}
