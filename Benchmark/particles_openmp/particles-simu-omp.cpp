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

#include "Scheduler/SpMultiPrioScheduler.hpp"

#include <omp.h>


class ParticlesGroup{
public:
    enum ValueTypes{
        PHYSICAL,
        X,
        Y,
        Z,
        FX,
        FY,
        FZ,
        POTENTIAL,
        NB_VALUE_TYPES
    };

private:
    std::size_t nbParticles;
    std::vector<double> values[NB_VALUE_TYPES];

public:
    explicit ParticlesGroup(const std::size_t inNbParticles = 0)
        : nbParticles(inNbParticles){
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[idxValueType].resize(nbParticles, 0);
        }
    }

    ParticlesGroup(const ParticlesGroup&) = default;
    ParticlesGroup(ParticlesGroup&&) = default;
    ParticlesGroup& operator=(const ParticlesGroup&) = default;
    ParticlesGroup& operator=(ParticlesGroup&&) = default;

    void setParticleValues(const std::size_t inIdxParticle,
                           const std::array<double, NB_VALUE_TYPES>& inValues){
        assert(inIdxParticle < nbParticles);
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[idxValueType][inIdxParticle] = inValues[idxValueType];
        }
    }

    auto getParticle(const std::size_t inIdxParticle) const {
        std::array<double, NB_VALUE_TYPES> valuesPart;
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            valuesPart[idxValueType] = values[idxValueType][inIdxParticle];
        }
        return valuesPart;
    }

    auto getNbParticles() const{
        return nbParticles;
    }

    void computeSelf(){
        for(std::size_t idxTarget = 0 ; idxTarget < getNbParticles() ; ++idxTarget){
            const double tx = double(values[X][idxTarget]);
            const double ty = double(values[Y][idxTarget]);
            const double tz = double(values[Z][idxTarget]);
            const double tv = double(values[PHYSICAL][idxTarget]);

            double  tfx = double(0.);
            double  tfy = double(0.);
            double  tfz = double(0.);
            double  tpo = double(0.);

            for(std::size_t idxSource = idxTarget+1  ; idxSource < nbParticles ; ++idxSource){
                double dx = tx - double(values[X][idxSource]);
                double dy = ty - double(values[Y][idxSource]);
                double dz = tz - double(values[Z][idxSource]);

                double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                const double inv_distance = sqrt(inv_square_distance);

                inv_square_distance *= inv_distance;
                inv_square_distance *= tv * double(values[PHYSICAL][idxSource]);

                dx *= - inv_square_distance;
                dy *= - inv_square_distance;
                dz *= - inv_square_distance;

                tfx += dx;
                tfy += dy;
                tfz += dz;
                tpo += inv_distance * double(values[PHYSICAL][idxSource]);

                values[FX][idxSource] -= dx;
                values[FY][idxSource] -= dy;
                values[FZ][idxSource] -= dz;
                values[POTENTIAL][idxSource] += inv_distance * tv;
            }

            values[FX][idxTarget] += tfx;
            values[FY][idxTarget] += tfy;
            values[FZ][idxTarget] += tfz;
            values[POTENTIAL][idxTarget] += tpo;
        }
    }

    void compute(ParticlesGroup& inOther){
        for(std::size_t idxTarget = 0 ; idxTarget < inOther.getNbParticles() ; ++idxTarget){
            const double tx = double(inOther.values[X][idxTarget]);
            const double ty = double(inOther.values[Y][idxTarget]);
            const double tz = double(inOther.values[Z][idxTarget]);
            const double tv = double(inOther.values[PHYSICAL][idxTarget]);

            double  tfx = double(0.);
            double  tfy = double(0.);
            double  tfz = double(0.);
            double  tpo = double(0.);

            for(std::size_t idxSource = 0  ; idxSource < nbParticles ; ++idxSource){
                double dx = tx - double(values[X][idxSource]);
                double dy = ty - double(values[Y][idxSource]);
                double dz = tz - double(values[Z][idxSource]);

                double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                const double inv_distance = sqrt(inv_square_distance);

                inv_square_distance *= inv_distance;
                inv_square_distance *= tv * double(values[PHYSICAL][idxSource]);

                dx *= - inv_square_distance;
                dy *= - inv_square_distance;
                dz *= - inv_square_distance;

                tfx += dx;
                tfy += dy;
                tfz += dz;
                tpo += inv_distance * double(values[PHYSICAL][idxSource]);

                values[FX][idxSource] -= dx;
                values[FY][idxSource] -= dy;
                values[FZ][idxSource] -= dz;
                values[POTENTIAL][idxSource] += inv_distance * tv;
            }

            inOther.values[FX][idxTarget] += tfx;
            inOther.values[FY][idxTarget] += tfy;
            inOther.values[FZ][idxTarget] += tfz;
            inOther.values[POTENTIAL][idxTarget] += tpo;
        }
    }

    /////////////////////////////////////////////////////////////

    #ifdef SPECX_COMPILE_WITH_CUDA  
    struct DataDescr {
        std::size_t nbParticles;
        double* values;

        double* getRawPtr() const{
            return values;
        }

        size_t getRawSize() const{
            return nbParticles*sizeof(float)*NB_VALUE_TYPES;
        }
    };

    auto memmovHostToDevice(cudaStream_t stream){
        double* doubleDevicePtr;
        CUDA_ASSERT(cudaMalloc(&doubleDevicePtr, nbParticles*NB_VALUE_TYPES*sizeof(double)));
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            CUDA_ASSERT(cudaMemcpyAsync(&doubleDevicePtr[idxValueType*nbParticles], values[idxValueType].data(), nbParticles*sizeof(double), cudaMemcpyHostToDevice, stream));
        }
        return DataDescr{nbParticles, doubleDevicePtr};
    }

    void memmovDeviceToHost(DataDescr& inDescr, cudaStream_t stream){
        double* doubleDevicePtr = inDescr.values;
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            CUDA_ASSERT(cudaMemcpyAsync(values[idxValueType].data(), &doubleDevicePtr[idxValueType*nbParticles], nbParticles*sizeof(double), cudaMemcpyDeviceToHost, stream));
        }
        CUDA_ASSERT(cudaFreeAsync(doubleDevicePtr, stream));
    }
    #endif
};

#if defined(SPECX_COMPILE_WITH_CUDA) || defined(SPECX_COMPILE_WITH_HIP)
template <class T>
__device__ T CuMin(const T& v1, const T& v2){
    return v1 < v2 ? v1 : v2;
}

__global__ void p2p_inner_gpu(void* data, std::size_t size){
    const std::size_t nbParticles = size/sizeof(double)/ParticlesGroup::NB_VALUE_TYPES;
    double* values[ParticlesGroup::NB_VALUE_TYPES];
    for(std::size_t idxValueType = 0 ; idxValueType < ParticlesGroup::NB_VALUE_TYPES ; ++idxValueType){
        values[idxValueType] = reinterpret_cast<double*>(data)+idxValueType*nbParticles;
    }

    constexpr std::size_t SHARED_MEMORY_SIZE = 128;
    const std::size_t nbThreads = blockDim.x*gridDim.x;
    const std::size_t uniqueId = threadIdx.x + blockIdx.x*blockDim.x;
    const std::size_t nbIterations = ((nbParticles+nbThreads-1)/nbThreads)*nbThreads;

    for(std::size_t idxTarget = uniqueId ; idxTarget < nbIterations ; idxTarget += nbThreads){
        const bool threadCompute = (idxTarget<nbParticles);

        double tx;
        double ty;
        double tz;
        double tv;

        if(threadCompute){
            tx = double(values[ParticlesGroup::X][idxTarget]);
            ty = double(values[ParticlesGroup::Y][idxTarget]);
            tz = double(values[ParticlesGroup::Z][idxTarget]);
            tv = double(values[ParticlesGroup::PHYSICAL][idxTarget]);
        }

        double  tfx = double(0.);
        double  tfy = double(0.);
        double  tfz = double(0.);
        double  tpo = double(0.);

        for(std::size_t idxCopy = 0 ; idxCopy < nbParticles ; idxCopy += SHARED_MEMORY_SIZE){
            __shared__ double sourcesX[SHARED_MEMORY_SIZE];
            __shared__ double sourcesY[SHARED_MEMORY_SIZE];
            __shared__ double sourcesZ[SHARED_MEMORY_SIZE];
            __shared__ double sourcesPhys[SHARED_MEMORY_SIZE];

            const std::size_t nbCopies = CuMin(SHARED_MEMORY_SIZE, nbParticles-idxCopy);
            for(std::size_t idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                sourcesX[idx] = values[ParticlesGroup::X][idx+idxCopy];
                sourcesY[idx] = values[ParticlesGroup::Y][idx+idxCopy];
                sourcesZ[idx] = values[ParticlesGroup::Z][idx+idxCopy];
                sourcesPhys[idx] = values[ParticlesGroup::PHYSICAL][idx+idxCopy];
            }

            __syncthreads();

            if(threadCompute){
                for(std::size_t otherIndex = 0; otherIndex < nbCopies; ++otherIndex) {
                    if(idxCopy + otherIndex != idxTarget){
                        double dx = tx - sourcesX[otherIndex];
                        double dy = ty - sourcesY[otherIndex];
                        double dz = tz - sourcesZ[otherIndex];

                        double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                        const double inv_distance = sqrt(inv_square_distance);

                        inv_square_distance *= inv_distance;
                        inv_square_distance *= tv * sourcesPhys[otherIndex];

                        dx *= - inv_square_distance;
                        dy *= - inv_square_distance;
                        dz *= - inv_square_distance;

                        tfx += dx;
                        tfy += dy;
                        tfz += dz;
                        tpo += inv_distance * sourcesPhys[otherIndex];
                    }
                }
            }

            __syncthreads();
        }

        if( threadCompute ){
            values[ParticlesGroup::FX][idxTarget] += tfx;
            values[ParticlesGroup::FY][idxTarget] += tfy;
            values[ParticlesGroup::FZ][idxTarget] += tfz;
            values[ParticlesGroup::POTENTIAL][idxTarget] += tpo;

        }

        __syncthreads();
    }
}

__global__ void p2p_neigh_gpu(const void* dataSrc, std::size_t sizeSrc,
                              void* dataTgt, std::size_t sizeTgt){
    const std::size_t nbParticlesTgt = sizeTgt/sizeof(double)/ParticlesGroup::NB_VALUE_TYPES;
    double* valuesTgt[ParticlesGroup::NB_VALUE_TYPES];
    for(std::size_t idxValueType = 0 ; idxValueType < ParticlesGroup::NB_VALUE_TYPES ; ++idxValueType){
        valuesTgt[idxValueType] = reinterpret_cast<double*>(dataTgt)+idxValueType*nbParticlesTgt;
    }

    const std::size_t nbParticlesSrc = sizeSrc/sizeof(double)/ParticlesGroup::NB_VALUE_TYPES;
    const double* valuesSrc[ParticlesGroup::NB_VALUE_TYPES];
    for(std::size_t idxValueType = 0 ; idxValueType < ParticlesGroup::NB_VALUE_TYPES ; ++idxValueType){
        valuesSrc[idxValueType] = reinterpret_cast<const double*>(dataSrc)+idxValueType*nbParticlesSrc;
    }

    constexpr std::size_t SHARED_MEMORY_SIZE = 128;
    const std::size_t nbThreads = blockDim.x*gridDim.x;
    const std::size_t uniqueId = threadIdx.x + blockIdx.x*blockDim.x;
    const std::size_t nbIterations = ((nbParticlesTgt+nbThreads-1)/nbThreads)*nbThreads;

    for(std::size_t idxTarget = uniqueId ; idxTarget < nbIterations ; idxTarget += nbThreads){
        const bool threadCompute = (idxTarget<nbParticlesTgt);

        double tx;
        double ty;
        double tz;
        double tv;

        if(threadCompute){
            tx = double(valuesTgt[ParticlesGroup::X][idxTarget]);
            ty = double(valuesTgt[ParticlesGroup::Y][idxTarget]);
            tz = double(valuesTgt[ParticlesGroup::Z][idxTarget]);
            tv = double(valuesTgt[ParticlesGroup::PHYSICAL][idxTarget]);
        }

        double  tfx = double(0.);
        double  tfy = double(0.);
        double  tfz = double(0.);
        double  tpo = double(0.);

        for(std::size_t idxCopy = 0 ; idxCopy < nbParticlesSrc ; idxCopy += SHARED_MEMORY_SIZE){
            __shared__ double sourcesX[SHARED_MEMORY_SIZE];
            __shared__ double sourcesY[SHARED_MEMORY_SIZE];
            __shared__ double sourcesZ[SHARED_MEMORY_SIZE];
            __shared__ double sourcesPhys[SHARED_MEMORY_SIZE];

            const std::size_t nbCopies = CuMin(SHARED_MEMORY_SIZE, nbParticlesSrc-idxCopy);
            for(std::size_t idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                sourcesX[idx] = valuesSrc[ParticlesGroup::X][idx+idxCopy];
                sourcesY[idx] = valuesSrc[ParticlesGroup::Y][idx+idxCopy];
                sourcesZ[idx] = valuesSrc[ParticlesGroup::Z][idx+idxCopy];
                sourcesPhys[idx] = valuesSrc[ParticlesGroup::PHYSICAL][idx+idxCopy];
            }

            __syncthreads();

            if(threadCompute){
                for(std::size_t otherIndex = 0; otherIndex < nbCopies; ++otherIndex) {
                    double dx = tx - sourcesX[otherIndex];
                    double dy = ty - sourcesY[otherIndex];
                    double dz = tz - sourcesZ[otherIndex];

                    double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                    const double inv_distance = sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhys[otherIndex];

                    dx *= - inv_square_distance;
                    dy *= - inv_square_distance;
                    dz *= - inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhys[otherIndex];
                }
            }

            __syncthreads();
        }

        if( threadCompute ){
            valuesTgt[ParticlesGroup::FX][idxTarget] += tfx;
            valuesTgt[ParticlesGroup::FY][idxTarget] += tfy;
            valuesTgt[ParticlesGroup::FZ][idxTarget] += tfz;
            valuesTgt[ParticlesGroup::POTENTIAL][idxTarget] += tpo;
        }

        __syncthreads();
    }
}

#endif


#include <random>

template <class NumType = double>
void FillRandomValues(ParticlesGroup& inGroup, const int inSeed,
                      const NumType inPhysicalValue = 0.1,
                      const NumType inMinPos = 0, const NumType inMaxPos = 1){
    std::mt19937 gen(inSeed);
    std::uniform_real_distribution<> distrib(inMinPos, inMaxPos);

    std::array<double, ParticlesGroup::NB_VALUE_TYPES> vecPart;
    std::fill(vecPart.begin(), vecPart.end(), 0);
    vecPart[ParticlesGroup::PHYSICAL] = inPhysicalValue;

    for(std::size_t idxPart = 0 ; idxPart < inGroup.getNbParticles() ; ++idxPart){
        vecPart[ParticlesGroup::X] = distrib(gen);
        vecPart[ParticlesGroup::Y] = distrib(gen);
        vecPart[ParticlesGroup::Z] = distrib(gen);
        inGroup.setParticleValues(idxPart, vecPart);
    }
}

template <class ValueType = double>
ValueType ChechAccuracy(const ParticlesGroup& inGroup1, const ParticlesGroup& inGroup2){
    if(inGroup1.getNbParticles() != inGroup2.getNbParticles()){
        return std::numeric_limits<ValueType>::infinity();
    }

    ValueType maxDiff = 0;
    for(std::size_t idx = 0 ; idx < inGroup1.getNbParticles() ; ++idx){
        const auto p1 = inGroup1.getParticle(idx);
        const auto p2 = inGroup2.getParticle(idx);

        for(std::size_t idxVal = 0 ; idxVal < ParticlesGroup::NB_VALUE_TYPES ; ++idxVal){
            maxDiff = std::max(maxDiff, std::abs((p1[idxVal] - p2[idxVal])/(p1[idxVal] == 0? 1 : p1[idxVal])));
        }
    }
    return maxDiff;
}


struct TuneResult{
    int nbThreadsInner = 0;
    int nbBlocksInner = 0;
    int nbThreadsOuter = 0;
    int nbBlocksOuter = 0;
};


template <int MaxNbDevices, const bool FavorLocality>
auto GetPriority(const bool usePrioPairs, const int maxInteractions, const int minInteractions, const int currentInteractions){
    if(usePrioPairs){
        const int ratio = 5 * double(currentInteractions-minInteractions)/double(maxInteractions-minInteractions);
        const int invRatio = 5 - ratio;
        return SpPriority(SpMultiPrioScheduler<MaxNbDevices,FavorLocality>::GeneratePriorityWorkerPair(invRatio, ratio));
    }
    return SpPriority(0);
}


bool isCpuWorker;
#pragma omp threadprivate(isCpuWorker)


#ifdef SPECX_COMPILE_WITH_CUDA
// Cuda stream
cudaStream_t threadStream;
#pragma omp threadprivate(threadStream)
#endif

template <int MaxNbDevices, const bool FavorLocality>
auto BenchCore( const int NbLoops, const int MinPartsPerGroup, const int MaxPartsPerGroup,
                const int NbGroups, const int nbCpu, const int nbGpu, const TuneResult& inKernelConfig,
                const int maxInteractions, const int minInteractions){

    std::vector<ParticlesGroup> particleGroups(NbGroups);

    {
        std::random_device rd;
        std::uniform_int_distribution<int> dist(MinPartsPerGroup, MaxPartsPerGroup);
        for(auto& group : particleGroups){
            const int nbParts = dist(rd);
            group = ParticlesGroup(nbParts);
        }
    }  

    std::vector<double> minMaxAvg(3);
    minMaxAvg[0] = std::numeric_limits<double>::max();
    minMaxAvg[1] = std::numeric_limits<double>::min();
    minMaxAvg[2] = 0;

    const int nbStreams = 3;

    for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
        #pragma omp parallel num_threads(nbCpu + nbGpu*nbStreams)
        {
            isCpuWorker = true;
            #ifdef SPECX_COMPILE_WITH_CUDA
            if(nbCpu <= omp_get_thread_num() ){
                isCpuWorker = false;
                const int deviceId = (omp_get_thread_num() - nbCpu)/(nbStreams*SpCudaUtils::GetNbDevices());
                CUDA_ASSERT(cudaSetDevice(deviceId));
                CUDA_ASSERT(cudaStreamCreate(&threadStream));
            }
            #endif

            #pragma omp master
            {
                SpTimer timer;
                #pragma omp taskgroup
                {

                    for(int idxGroup1 = 0 ; idxGroup1 < int(particleGroups.size()) ; ++idxGroup1){
                        auto& group1 = particleGroups[idxGroup1];
                        const int priority = GetPriority<MaxNbDevices,FavorLocality>(false, maxInteractions, minInteractions, group1.getNbParticles()*group1.getNbParticles()).getPriority();
                        #pragma omp task depend(mutexinout:group1) firstprivate(group1) priority(priority) default(none)
                        {
                            if(isCpuWorker){
                                group1.computeSelf();
                            }
                            #ifdef SPECX_COMPILE_WITH_CUDA
                            else{
                                auto descr = group1.memmovHostToDevice(threadStream);
                                p2p_inner_gpu<<<inKernelConfig.nbBlocksInner,inKernelConfig.nbThreadsInner,0,threadStream>>>
                                                 (descr.getRawPtr(), descr.getRawSize());
                                group1.memmovDeviceToHost(descr, threadStream);
                                CUDA_ASSERT(cudaStreamSynchronize(threadStream));
                            }
                            #endif
                        }
                    }

                    for(int idxGroup1 = 0 ; idxGroup1 < int(particleGroups.size()) ; ++idxGroup1){
                        auto& group1 = particleGroups[idxGroup1];

                        const int idxGroup2 = (idxGroup1+1)%particleGroups.size();
                        auto& group2 = particleGroups[idxGroup2];

                        const int priority = GetPriority<MaxNbDevices,FavorLocality>(false, maxInteractions, minInteractions, group1.getNbParticles()*group2.getNbParticles()).getPriority();

                        #pragma omp task depend(mutexinout:group1, group2) firstprivate(group1, group2) priority(priority) default(none)
                        {
                            if(isCpuWorker){
                                group1.compute(group2);
                            }
                            #ifdef SPECX_COMPILE_WITH_CUDA
                            else{
                                auto descr1 = group1.memmovHostToDevice(threadStream);
                                auto descr2 = group2.memmovHostToDevice(threadStream);
                                p2p_neigh_gpu<<<inKernelConfig.nbBlocksOuter,inKernelConfig.nbThreadsOuter,0,threadStream>>>
                                                (descr2.getRawPtr(), descr2.getRawSize(), descr1.getRawPtr(), descr1.getRawSize());
                                p2p_neigh_gpu<<<inKernelConfig.nbBlocksOuter,inKernelConfig.nbThreadsOuter,0,threadStream>>>
                                                (descr1.getRawPtr(), descr1.getRawSize(), descr2.getRawPtr(), descr2.getRawSize());
                                group1.memmovDeviceToHost(descr1, threadStream);
                                group2.memmovDeviceToHost(descr2, threadStream);
                                CUDA_ASSERT(cudaStreamSynchronize(threadStream));
                            }
                            #endif
                        }
                    }
                }

                timer.stop();

                minMaxAvg[0] = std::min(minMaxAvg[0], timer.getElapsed());
                minMaxAvg[1] = std::max(minMaxAvg[1], timer.getElapsed());
                minMaxAvg[2] += timer.getElapsed();
            }

            #pragma omp barrier

            #ifdef SPECX_COMPILE_WITH_CUDA
            if(isCpuWorker == false){
                CUDA_ASSERT(cudaStreamDestroy(threadStream));
            }
            #endif
        }
    }

    minMaxAvg[2] /= NbLoops;

    return minMaxAvg;
}

void BenchmarkTest(int argc, char** argv, const TuneResult& inKernelConfig){
    CLsimple args("Particles", argc, argv);

    args.addParameterNoArg({"help"}, "help");

    int NbLoops = 100;
    args.addParameter<int>({"lp" ,"nbloops"}, "NbLoops", NbLoops, 100);

    int MinPartsPerGroup;
    args.addParameter<int>({"minp"}, "MinPartsPerGroup", MinPartsPerGroup, 100);

    int MaxPartsPerGroup;
    args.addParameter<int>({"maxp"}, "MaxPartsPerGroup", MaxPartsPerGroup, 300);

    int MinNbGroups;
    args.addParameter<int>({"minnbgroups"}, "MinNbGroups", MinNbGroups, 10);

    int MaxNbGroups;
    args.addParameter<int>({"maxnbgroups"}, "MaxNbGroups", MaxNbGroups, 10);

    std::string outputDir = "./";
    args.addParameter<std::string>({"od"}, "outputdir", outputDir, outputDir);

    args.parse();

    if(!args.isValid() || args.hasKey("help")){
      // Print the help
      args.printHelp(std::cout);
      return;
    }

    const int maxInteractions = MaxPartsPerGroup*MaxPartsPerGroup;
    const int minInteractions = MinPartsPerGroup*MinPartsPerGroup;

    assert(MinPartsPerGroup <= MaxPartsPerGroup);
    assert(MinNbGroups <= MaxNbGroups);

#ifdef SPECX_COMPILE_WITH_CUDA  
    SpCudaUtils::PrintInfo(); 
    const int nbMaxGpus = SpCudaUtils::GetNbDevices();
#else    
    const int nbMaxGpus = 0;
#endif 

    std::vector<std::vector<double>> allDurations;

    for(int idxGpu = -1 ; idxGpu <= nbMaxGpus ; ++idxGpu){
        for(int idxBlock = MinNbGroups ; idxBlock <= MaxNbGroups ; idxBlock *= 2){
            const int nbGpus = (idxGpu == -1 ? 0 : idxGpu);
            const int nbCpus = std::max(1 , (idxGpu == -1 ? 1 : SpUtils::DefaultNumThreads())-nbGpus);

            std::cout << "NbGpu = " << idxGpu << " BlockSize = " << idxBlock << std::endl;              
            const auto minMaxAvg = BenchCore<8,true>(NbLoops, MinPartsPerGroup,
                                                    MaxPartsPerGroup, idxBlock, nbCpus, nbGpus, inKernelConfig,
                                                    maxInteractions, minInteractions);
            allDurations.push_back(minMaxAvg);
            std::cout << " - Min = " << minMaxAvg[0] << " Max = " << minMaxAvg[1] << " Avg = " << minMaxAvg[2] << std::endl;  
        }
    }

    std::ofstream file(outputDir + "/particle-simu-omp.csv");
    if(!file.is_open()){
        std::cerr << "Cannot open file " << outputDir + "/particle-simu-omp.csv" << std::endl;
        return;
    }

    file << "NbCpu,NbGpu,BlockSize,MinDuration,MaxDuration,AvgDuration" << std::endl;
    int idxDuration = 0;
    for(int idxGpu = -1 ; idxGpu <= nbMaxGpus ; ++idxGpu){
        for(int idxBlock = MinNbGroups ; idxBlock <= MaxNbGroups ; idxBlock *= 2){
            const int nbCpus = (idxGpu == -1 ? 1 : SpUtils::DefaultNumThreads());
            const int nbGpus = (idxGpu == -1 ? 0 : idxGpu);

            file << nbCpus << "," << nbGpus << "," << idxBlock << "," 
                    << allDurations[idxDuration][0] << "," 
                    << allDurations[idxDuration][1] << "," 
                    << allDurations[idxDuration][2] << std::endl;
            idxDuration += 1;
        }
    }
}


int main(int argc, char** argv){

    TuneResult tuneConfig{256, 128, 256, 128};
    // Test if it is in a file "tuneConfig.txt"
    {
        std::ifstream file("tuneConfig.txt");
        if(file.is_open()){
            file >> tuneConfig.nbThreadsInner;
            file >> tuneConfig.nbBlocksInner;
            file >> tuneConfig.nbThreadsOuter;
            file >> tuneConfig.nbBlocksOuter;
            std::cout << "Tune config loaded from file" << std::endl;
            std::cout << " - Inner block " << tuneConfig.nbBlocksInner << " threads " << tuneConfig.nbThreadsInner << std::endl;
            std::cout << " - Outer block " << tuneConfig.nbBlocksOuter << " threads " << tuneConfig.nbThreadsOuter << std::endl;
        }
    }

    BenchmarkTest(argc, argv, tuneConfig);

    return 0;
}
