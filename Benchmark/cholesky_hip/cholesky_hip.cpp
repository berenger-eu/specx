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
            return sizeof(double)*dimension;
        }

         template <class DeviceMemmov>
            auto memmovHostToDevice(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size){
                assert(size == sizeof(double)*dimension);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyHostToDevice(doubleDevicePtr, data, sizeof(double)*dimension);
                return DataDescr(dimension);
            }

        template <class DeviceMemmov>
            void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr,[[maybe_unused]] std::size_t size, const DataDescr& /*inDataDescr*/){
                //int dim=num_columns*num_rows;
                assert(size == sizeof(double)*dimension);
                double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
                mover.copyDeviceToHost(data, doubleDevicePtr, sizeof(double)*dimension);
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
	if (idx < r*c) { i=idx/r; j=idx-i*r;  if (j<i) { R[i * r + j]; } }
}

#endif


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

Matrix build_init_matrix(unsigned int num_rows, unsigned int num_columns)
{
	Matrix M;
	M.num_columns = M.pitch = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
    M.dimension=M.num_rows * M.num_columns;
	M.data = (double *)malloc(size * sizeof(double));

	// Step 1: Create a matrix with random numbers between [-.5 and .5]
    std::cout<<"[INFO]: Create Matrix definite positiv"<<"\n";
    std::cout<<"[INFO]: Creating a"<<num_rows<<"x"<<num_columns<<"matrix with random numbers between [-.5, .5]... ";
	unsigned int i;
	unsigned int j;
	for(i = 0; i < size; i++)
		M.data[i] = ((double)rand()/(double)RAND_MAX) - 0.5;
        std::cout<<"done"<<"\n";

	// Step 2: Make the matrix symmetric by adding its transpose to itself
    std::cout<<"[INFO]: Generating the symmetric matrix...";
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
		std::cout<<"done"<<"\n";
	else{ 
        std::cout<<"error !!!"<<"\n";
		free(M.data);
		M.data = NULL;
	}
	// Step 3: Make the diagonal entries large with respect to the row and column entries
    std::cout<<"Generating the positive definite matrix...";
	for(i = 0; i < num_rows; i++)
		for(j = 0; j < num_columns; j++){
			if(i == j) 
				M.data[i * M.num_rows + j] += 0.5 * M.num_rows;
		}
	if(check_if_diagonal_dominant(M))
		std::cout<<"done"<<"\n";
	else{
		std::cout<<"error !!!"<<"\n";
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
    int size = M.dimension;

    M.data = (double *) malloc(size * sizeof (double));
    for (unsigned int i = 0; i < size; i++) {
        if (init == 0) M.data[i] = 0;
        else
            M.data[i] = (double) rand() / (double) RAND_MAX;
    }
    return M;
}

Matrix allocate_matrix_on_gpu(const Matrix M){
    Matrix Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(double);
    hipMalloc((void**)&Mdevice.data, size);
    return Mdevice;
}


void copy_matrix_to_device(Matrix Mdevice, const Matrix Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(double);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    Mdevice.pitch = Mhost.pitch;
    hipMemcpy(Mdevice.data, Mhost.data, size, hipMemcpyHostToDevice);
}

void copy_matrix_from_device(Matrix Mhost, const Matrix Mdevice){
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(double);
    hipMemcpy(Mhost.data, Mdevice.data, size, hipMemcpyDeviceToHost);
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
	Matrix R= allocate_matrix(A.num_columns,A.num_rows,0);

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

	hipLaunchKernelGGL(matrix_transpose,grid, thread_block,0,0,gpu_R.data,gpu_A.data,A.num_rows,A.num_columns); 

	copy_matrix_from_device(R,gpu_R);
	hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
	hipFree(gpu_A.data);
	hipFree(gpu_R.data);
	return R;
}

Matrix matrix_copy_GPU(const Matrix A) 
{
	int block_size = 512;
    //Matrix R= allocate_matrix(A.num_rows,A.num_columns,0);
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

bool is_matrix_equal_GPU(const Matrix A, const Matrix B) 
{
	double deltaError=0.000001;
	return(is_matrix_equal_GPU(A,B,deltaError));
}


void checkSolution_GPU(Matrix A,Matrix B)
{
	bool res=is_matrix_equal_GPU(A,B);
	printf("[INFO]:	%s\n", (true == res) ? "WELL DONE PASSED :-)" : "FAILED");
}




void BenchmarkTest(int argc, char** argv){
    // ./axpy_hip --sz=10 --th=256
    std::cout<<"[INFO]: BENCHMARK Cholesky Resolution"<<"\n";

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
    bool qView=true;
    //qView=false;	

    Matrix MatA; MatA = build_init_matrix(matrixSize,matrixSize); 
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
                            const int size = paramA.getRawSize()/sizeof(double);
                            int matrixSize=sqrt(size);
                            int threads_per_block = nbthreads;
                            int stride            = threads_per_block;
                            //std::cout<<"*********** Nbdata="<<size<<"\n";
                            //std::cout<<"*********** matrixSize="<<matrixSize<<"\n";
                            int k;
                            for (k = 0; k < matrixSize; k++) {
                                int isize = (matrixSize - 1) - (k + 1) + 1;
                                int num_blocks = isize;
                                if (num_blocks <= 0) { num_blocks = 1; }
                                dim3 thread_block(threads_per_block, 1, 1);
                                dim3 grid(num_blocks, 1);
                                hipLaunchKernelGGL(chol_kernel_optimized_div,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                                hipLaunchKernelGGL(chol_kernel_optimized,grid,thread_block,0,SpHipUtils::GetCurrentStream(),(double*)paramU.getRawPtr(),k,stride,matrixSize); 
                            }

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
    matrix_lower_triangular(MatU);
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


int main(int argc, char** argv){
    std::cout<<"\n";
    BenchmarkTest(argc, argv);
    std::cout<<"[INFO]: FINISHED..."<<"\n";
    return 0;
}
