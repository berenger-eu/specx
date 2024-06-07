#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

#pragma GCC diagnostic warning "-Wunused-result"
#pragma clang diagnostic ignored "-Wunused-result"

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
#include <hipblas.h>
#include <hiprand.h>
 


struct Matrix {
 	  unsigned int num_columns;
   	  unsigned int num_rows;
      unsigned int dimension; 
      unsigned int dimensionSizeof; 
 	  unsigned int pitch; 
 	  double* data;

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
            return dimensionSizeof;
        }

        template <class DeviceMemmov>
            auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
                assert(size == dimensionSizeof);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyHostToDevice(doubleDevicePtr, data, dimensionSizeof);
                return DataDescr(dimension);
            }

        template <class DeviceMemmov>
            void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
                assert(size == dimensionSizeof);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyDeviceToHost(data, doubleDevicePtr, dimensionSizeof);
            }
};



#ifdef SPECX_COMPILE_WITH_HIP

__global__ void chol_kernel_optimized_div(double * U, int k, int stride, int m) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j;
    unsigned int num_rows = m;
    if (tx == 0) { U[k * num_rows + k] = sqrt(U[k * num_rows + k]); }
    int offset  = (k + 1); 
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = (k + 1);
    if (blockIdx.x == 0) {
        for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
            U[k * num_rows + j] /= U[k * num_rows + k]; 
        }
    }
}

__global__ void chol_kernel_optimized(double * U, int k, int stride, int m) {
    unsigned int j;
    unsigned int num_rows = m; 
    int i       = blockIdx.x + (k + 1);
    int offset  = i;
    int jstart  = threadIdx.x + offset;
    int jstep   = stride;
    int jtop    = num_rows - 1;
    int jbottom = i;
    for (j = jstart; (j >= jbottom) && (j <= jtop); j += jstep) {
        U[i * num_rows + j] -= U[k * num_rows + i] * U[k * num_rows + j];
    }
}

__global__ void matrix_mult(double* C, double* A, double* B, int m, int n) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	int i = idx / n;
	int k = idx - n * i;
	if (n * m > idx) {
		for (int j = 0; j < n; j++) {
			C[idx] += A[n * i + j] * B[n * j + k];
		}
	}
}
 
__global__ void matrix_equal(volatile bool *Q, double* A, double* B, int nb, double deltaError) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	if (idx < nb)
		if (abs(A[idx]-B[idx])>deltaError) { Q[0]=false;  } 
}

__global__ void matrix_copy(double *R, double *A, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  R[j * c + i] = A[j * c + i]; }
}

__global__ void matrix_transpose(double *R, double *A, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  R[i * r + j] = A[j * c + i]; }
}

__global__ void matrix_lower_triangular(double *R, int r, int c) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 
	unsigned int i,j;
	if (idx < r*c) { i=idx/r; j=idx-i*r;  if (j<i) { R[i * r + j]=0.0; } }
}


#endif




//BEGIN::Tools for Matrix manipulation

void writeMatrix(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++){
		for(unsigned int j = 0; j < M.num_columns; j++)
		{
			printf("%f ", M.data[i*M.num_columns + j]);
		}
		printf("\n");
	} 
	printf("\n");
}


int check_if_symmetric(const Matrix M)
{
	for(unsigned int i = 0; i < M.num_rows; i++)
		for(unsigned int j = 0; j < M.num_columns; j++)
			if(M.data[i * M.num_rows + j] != M.data[j * M.num_columns + i]) return 0;
	return 1;
}

int check_if_diagonal_dominant(const Matrix M)
{
	float diag_element;
	float sum;
	for(unsigned int i = 0; i < M.num_rows; i++){
		sum = 0.0; 
		diag_element = M.data[i * M.num_rows + i];
		for(unsigned int j = 0; j < M.num_columns; j++){
			if(i != j) sum += abs(M.data[i * M.num_rows + j]);
		}
		if(diag_element <= sum) return 0;
	}
	return 1;
}

Matrix build_init_matrix(unsigned int num_rows, unsigned int num_columns,bool qView)
{
	Matrix M;
	M.num_columns     = M.pitch = num_columns;
	M.num_rows        = num_rows; 
    unsigned int size = M.num_rows * M.num_columns;
    M.dimension       = size;
    M.dimensionSizeof = size*sizeof(double);
	M.data            = (double *)malloc(M.dimensionSizeof);

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
    if (qView) { std::cout<<"[INFO]: Create Matrix definite positiv"<<"\n"; }
    if (qView) { std::cout<<"[INFO]: Creating a "<<num_rows<<"x"<<num_columns<<" matrix with random numbers between [-.5, .5]... "; }
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.data[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
        if (qView) { std::cout<<"done"<<"\n"; }

	// Step 2: Make the matrix symmetric by adding its transpose to itself
    if (qView) { std::cout<<"[INFO]: Generating the symmetric matrix... ";}
	Matrix transpose;
	transpose.num_columns = transpose.pitch = num_columns;
	transpose.num_rows = num_rows; 
	size = transpose.num_rows * transpose.num_columns;
	transpose.data = (double *)malloc(size * sizeof(double));

	for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			transpose.data[i * M.num_rows + j] = M.data[j * M.num_columns + i];
	// writeMatrix(transpose);

	for(i = 0; i < size; i++)
		M.data[i] += transpose.data[i];

	if (check_if_symmetric(M))
	{ 
        if (qView) { std::cout<<"done"<<"\n"; } 
    }
	else
    { 
        if (qView) {std::cout<<"error !!!"<<"\n"; }
		free(M.data);
		M.data = NULL;
	}
	// Step 3: Make the diagonal entries large with respect to the row and column entries
    if (qView) {  std::cout<<"[INFO]: Generating the positive definite matrix... "; }
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++)
        {
			if(i == j) 
				M.data[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if (check_if_diagonal_dominant(M))
    {
		if (qView) { std::cout<<"done"<<"\n"; }
    }
	else
    {
		if (qView) { std::cout<<"error !!!"<<"\n"; }
		free(M.data);
		M.data = NULL;
	}
	free(transpose.data);
	return M;
}

Matrix allocate_matrix(int num_rows, int num_columns, int init) {
    Matrix M;
    M.num_columns = M.pitch = num_columns;
    M.num_rows = num_rows;
    M.dimension=M.num_rows * M.num_columns;
    unsigned int size = M.num_rows * M.num_columns;
    M.dimension=size;
    M.dimensionSizeof=size*sizeof(double);
	M.data = (double *)malloc(M.dimensionSizeof);
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.data[i] = 0;
        else
            M.data[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}

Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    hipMalloc((void**)&Mdevice.data, M.dimensionSizeof);
    return Mdevice;
}

void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    hipMemcpy(Mdevice.data, Mhost.data,Mhost.dimensionSizeof,hipMemcpyHostToDevice);
}

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    hipMemcpy(Mhost.data, Mdevice.data,Mdevice.dimensionSizeof,hipMemcpyDeviceToHost);
}

void matrix_lower_triangular(Matrix M) 
{
    int i, j;
    for (i = 0; i < M.num_rows; i++)
        for (j = 0; j < i; j++)
            M.data[i * M.num_rows + j] = 0.0;
}

Matrix matrix_copy(const Matrix M) 
{
  Matrix R= allocate_matrix(M.num_rows,M.num_columns,0);
  int i,j;
  for(i = 0; i < M.num_rows; i++)
		for(j = 0; j < M.num_columns; j++)
			R.data[i * M.num_rows + j] = M.data[i * M.num_rows + j];
  return R;
}
 
//END::Tools for Matrix manipulation


//BEGIN::Tools for Matrix manipulation in GPU

#ifdef SPECX_COMPILE_WITH_HIP

Matrix matrix_product_GPU(const Matrix A, const Matrix B) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
    Matrix C= allocate_matrix(matrixSize,matrixSize,0);

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	Matrix gpu_C = allocate_matrix_on_gpu(C);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	copy_matrix_to_device(gpu_C, C );
	
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_mult,grid, thread_block,0,0,gpu_C.data,gpu_A.data,gpu_B.data,matrixSize,matrixSize); 

	copy_matrix_from_device(C,gpu_C);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_B.data);
	hipFree(gpu_C.data);

	return C;
}

Matrix matrix_transpose_GPU(const Matrix A) 
{
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
	Matrix R= allocate_matrix(A.num_columns,A.num_rows,0); //<== Transpose Matrix

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_R = allocate_matrix_on_gpu(R);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_R, R );
	
	int num_blocks = (A.dimension + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_transpose,grid, thread_block,0,0,gpu_R.data,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(R,gpu_R);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_R.data);
	return R;
}

Matrix matrix_lower_triangular_GPU(const Matrix A) 
{
	int block_size = 512;
	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	
	int num_blocks = (A.dimension + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_lower_triangular,grid, thread_block,0,0,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(A,gpu_A);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	return A;
}

Matrix matrix_copy_GPU(const Matrix A) 
{
	int block_size = 512;
	Matrix R= allocate_matrix(A.num_rows,A.num_columns,0); 

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_R = allocate_matrix_on_gpu(R);

	hipEventRecord(start, 0);   

	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_R, R );
	
	int num_blocks = (A.num_columns*A.num_rows + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);

	hipLaunchKernelGGL(matrix_copy,grid, thread_block,0,0,gpu_R.data,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(R,gpu_R);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_R.data);
	return R;
}



bool is_matrix_equal_GPU(const Matrix A, const Matrix B,const double deltaError) 
{
	int block_size = 512;
	int matrixSize=A.num_columns;
	int sizeQ = sizeof(bool) * 1;
    bool *h_Q = (bool *)malloc(sizeQ);
	h_Q[0]=true;

	hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

	Matrix gpu_A = allocate_matrix_on_gpu(A);
	Matrix gpu_B = allocate_matrix_on_gpu(B);
	bool  *d_Q;    hipMalloc((void **)&d_Q,sizeQ);

	hipEventRecord(start, 0);   
	copy_matrix_to_device(gpu_A, A );
	copy_matrix_to_device(gpu_B, B );
	hipMemcpy(d_Q,h_Q,sizeQ, hipMemcpyHostToDevice);
	int num_blocks = (matrixSize*matrixSize + block_size - 1) / block_size;
	
	dim3 thread_block(block_size, 1, 1);
	dim3 grid(num_blocks, 1);
	hipLaunchKernelGGL(matrix_equal,grid, thread_block,0,0,d_Q,gpu_A.data,gpu_B.data,matrixSize*matrixSize,deltaError); 
	hipMemcpy(h_Q,d_Q,sizeof(bool), hipMemcpyDeviceToHost);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_B.data);
	hipFree(d_Q);

	return (h_Q[0]);
}

void checkSolution_GPU(Matrix A,Matrix B)
{
    const double deltaError=0.000001;
	bool res=is_matrix_equal_GPU(A,B,deltaError);
	printf("[INFO]:	%s\n", (true == res) ? "WELL DONE PASSED :-)" : "FAILED");
}

#endif

//END::Tools for Matrix manipulation in GPU






void BenchmarkTest(int argc, char** argv){
    // ./axpy_hip --sz=10 --th=256
    std::cout<<"[INFO]: BENCHMARK Cholesky Resolution"<<"\n";
    bool qView=true;
    //qView=false;	

    // BEGIN:: Init size vectors and Nb Threads in GPU HIP AMD
    CLsimple args("Cholesky", argc, argv);
    args.addParameterNoArg({"help"}, "help");
    int matrixSize=10;
    args.addParameter<int>({"sz" ,"size"}, "Size",matrixSize,10);
    int nbthreads;
    args.addParameter<int>({"th"}, "nbthreads", nbthreads, 256);
    args.parse();
    if(!args.isValid() || args.hasKey("help")){
      args.printHelp(std::cout);
      return;
    }
    // END:: Init size vectors and Nb Threads in GPU HIP AMD

    // BEGIN:: Init part
    std::chrono::steady_clock::time_point t_begin,t_end;
    long int t_laps;

    Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize,qView); 
    Matrix MatU=matrix_copy(MatA); 
    if (qView) { 
        std::cout<<"\n";
        std::cout << "Matrix A ===>\n\n";	writeMatrix(MatA); std::cout << "\n";
     }  
    // END:: Init part

    #ifdef SPECX_COMPILE_WITH_HIP
        std::cout<<"[INFO]: Start Hip Part..."<<"\n";
        t_begin = std::chrono::steady_clock::now();
        // BEGIN:: Task definition
        //SpHipUtils::PrintInfo();
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers());
        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,"should be stdvec");
        SpTaskGraph tg;
        tg.computeOn(ce);
        tg.task(SpCommutativeWrite(MatU),SpRead(MatA),
                SpHip([nbthreads](SpDeviceDataView<Matrix> paramU,
                        const SpDeviceDataView<const Matrix> paramA) {
                            //BEGIN:: Cholesky part
                                const int size = paramA.getRawSize()/sizeof(double);
                                const unsigned int matrixSize=sqrt(size);
                                const unsigned int threads_per_block = nbthreads;
                                const unsigned int stride            = threads_per_block;
                                unsigned int k;
                                for (k = 0; k < matrixSize; k++) {
                                    int isize = (matrixSize - 1) - (k + 1) + 1;
                                    int num_blocks = isize;
                                    if (num_blocks <= 0) { num_blocks = 1; }
                                    dim3 thread_block(threads_per_block, 1, 1);
                                    dim3 grid(num_blocks, 1);
                                    hipLaunchKernelGGL(chol_kernel_optimized_div,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                                    hipLaunchKernelGGL(chol_kernel_optimized,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                                }
                            //END:: Cholesky part

                            //BEGIN:: lower_triangular
                            const unsigned int block_size_2 = nbthreads;
                            const unsigned int num_blocks_2 = (size+ block_size_2 - 1) / block_size_2;
                            dim3 thread_block_2(block_size_2, 1, 1);
                            dim3 grid_2(num_blocks_2, 1);
                            hipLaunchKernelGGL(matrix_lower_triangular,grid_2,thread_block_2,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),matrixSize,matrixSize); 
                            //END:: lower_triangular
                })
        );

        tg.task(SpWrite(MatU), SpCpu([](Matrix&) { }));
        // END:: Task definition

        // BEGIN:: Task execution
        tg.waitAllTasks();
        //tg.stopIfNotAlreadyStopped();
        t_end = std::chrono::steady_clock::now();
        // END:: Task execution


        // BEGIN:: Show results
        std::cout<<"[INFO]: Generate trace..."<<"\n";
        tg.generateTrace("./Cholesky-hip.svg",true);

        std::cout << "\n";
        std::cout<<"[INFO]: Results..."<<"\n";

        std::cout<<"\n";
        if (qView) { 
            std::cout<<"\n";
            std::cout << "Matrix U ===>\n\n";	writeMatrix(MatU); std::cout << "\n";
        }

        t_laps= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
        std::cout << "[INFO]: Elapsed microseconds inside: "<<t_laps<< " us\n";
        std::cout << "\n";

        std::cout<<"[INFO]: CheckSolution"<<"\n";
        Matrix MatUt=matrix_transpose_GPU(MatU);
        Matrix MatT=matrix_product_GPU(MatUt,MatU);
        checkSolution_GPU(MatA,MatT);

        // END:: End results

        free(MatU.data);
        free(MatA.data);
    #endif
}


Matrix getCholeskyGPUVers3(Matrix A,int threads_per_block)
{
	int matrixSize=A.num_rows;
    Matrix U= allocate_matrix(matrixSize,matrixSize,0);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    //int threads_per_block = 256; 
    int stride = threads_per_block;    
    hipEventRecord(start, 0);    
    Matrix gpu_u = allocate_matrix_on_gpu(U);
    copy_matrix_to_device(gpu_u, A);
    int k;
    for (k = 0; k < matrixSize; k++) {
        int isize = (matrixSize - 1) - (k + 1) + 1;
        int num_blocks = isize;
        if (num_blocks <= 0) { num_blocks = 1; }
        dim3 thread_block(threads_per_block, 1, 1);
        dim3 grid(num_blocks, 1);
        hipLaunchKernelGGL(chol_kernel_optimized_div,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
        hipLaunchKernelGGL(chol_kernel_optimized,grid, thread_block,0,0,gpu_u.data,k,stride,matrixSize); 
    }

	const unsigned int block_size_2 = threads_per_block;
    const unsigned int num_blocks_2 = (A.dimension+ block_size_2 - 1) / block_size_2;
    dim3 thread_block_2(block_size_2, 1, 1);
    dim3 grid_2(num_blocks_2, 1);
	hipLaunchKernelGGL(matrix_lower_triangular,grid_2, thread_block_2,0,0,gpu_u.data,matrixSize,matrixSize);

    copy_matrix_from_device(U, gpu_u);  				 
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipFree(gpu_u.data);

	//matrix_lower_triangular(U);

    return U;
}

void TestPerform()
{
	std::ofstream myfile;
	myfile.open ("DataSpecx.csv");
    bool qView=true;
    qView=false;	
	bool qCTRL=true;
	qCTRL=false;

	myfile <<"DimMatrix,ModeSpecxHipAMD,ModeNormalGPU"<<"\n";	
	std::chrono::steady_clock::time_point t_begin,t_end;

    for (int i = 1; i <= 10; i++)
    {
        const int matrixSize=10000+i*(1000);
        int nbthreads=512;
        myfile <<matrixSize<<",";

        // BEGIN:: Init part
        std::chrono::steady_clock::time_point t_begin,t_end;
        long int t_laps_specx,t_laps_gpu;

        Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize,true); 
        Matrix MatU=matrix_copy(MatA); 
        if (qView) { 
            std::cout<<"\n";
            std::cout << "Matrix A ===>\n\n";	writeMatrix(MatA); std::cout << "\n";
        }  
        // END:: Init part

            std::cout<<"[INFO]: Start Hip Part..."<<"\n";
            sleep(1);
            t_begin = std::chrono::steady_clock::now();
            // BEGIN:: Task definition
            //SpHipUtils::PrintInfo();
            SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuHipWorkers());
            static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,"should be stdvec");
            SpTaskGraph tg;
            tg.computeOn(ce);
            tg.task(SpCommutativeWrite(MatU),SpRead(MatA),
                    SpHip([nbthreads](SpDeviceDataView<Matrix> paramU,
                            const SpDeviceDataView<const Matrix> paramA) {
                                //BEGIN:: Cholesky part
                                    const int size = paramA.getRawSize()/sizeof(double);
                                    const unsigned int matrixSize=sqrt(size);
                                    const unsigned int threads_per_block = nbthreads;
                                    const unsigned int stride            = threads_per_block;
                                    unsigned int k;
                                    for (k = 0; k < matrixSize; k++) {
                                        int isize = (matrixSize - 1) - (k + 1) + 1;
                                        int num_blocks = isize;
                                        if (num_blocks <= 0) { num_blocks = 1; }
                                        dim3 thread_block(threads_per_block, 1, 1);
                                        dim3 grid(num_blocks, 1);
                                        hipLaunchKernelGGL(chol_kernel_optimized_div,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                                        hipLaunchKernelGGL(chol_kernel_optimized,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                                    }
                                //END:: Cholesky part

                                //BEGIN:: lower_triangular
                                const unsigned int block_size_2 = nbthreads;
                                const unsigned int num_blocks_2 = (size+ block_size_2 - 1) / block_size_2;
                                dim3 thread_block_2(block_size_2, 1, 1);
                                dim3 grid_2(num_blocks_2, 1);
                                hipLaunchKernelGGL(matrix_lower_triangular,grid_2,thread_block_2,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),matrixSize,matrixSize); 
                                //END:: lower_triangular

                    })
            );

            tg.task(SpWrite(MatU), SpCpu([](Matrix&) { }));
            // END:: Task definition

            //matrix_lower_triangular(MatU);

            // BEGIN:: Task execution
            tg.waitAllTasks();
            //tg.stopIfNotAlreadyStopped();
            t_end = std::chrono::steady_clock::now();
            // END:: Task execution


            std::cout << "\n";
            std::cout << "\n";
            std::cout << "[INFO]: matrixSize= "<<matrixSize<<" x "<<matrixSize<<"\n";
            t_laps_specx= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            myfile<<t_laps_specx;
            myfile<<",";
            std::cout << "[INFO]: Elapsed microseconds inside SpecX  : "<<t_laps_specx<< " us\n";
            std::cout << "\n";

            sleep(1);
            t_begin = std::chrono::steady_clock::now();
            Matrix MatU_gpu2=getCholeskyGPUVers3(MatA,nbthreads);
            t_end = std::chrono::steady_clock::now();
            t_laps_gpu= std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count();
            myfile<<t_laps_gpu;
            std::cout << "[INFO]: Elapsed microseconds inside my GPU : "<<t_laps_gpu<< " us\n";
            std::cout << "\n";
            std::cout << "\n";


            //writeMatrix(MatU);
            //writeMatrix(MatU_gpu2);

            //std::cout<<"[INFO]: CheckSolution"<<"\n";
            //Matrix MatUt=matrix_transpose_GPU(MatU);
            //Matrix MatT=matrix_product_GPU(MatUt,MatU);
            //checkSolution_GPU(MatA,MatT);





            // END:: End results
            free(MatU_gpu2.data);
            free(MatU.data);
            free(MatA.data);
            myfile<<"\n";

            sleep(1);


    }

    myfile.close();

}


int main(int argc, char** argv){
    std::cout<<"\n";
    //BenchmarkTest(argc, argv);
    TestPerform();
    std::cout<<"[INFO]: FINISHED..."<<"\n";
    return 0;
}
