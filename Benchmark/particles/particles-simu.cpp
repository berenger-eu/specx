///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"
#include "Task/SpTask.hpp"

#include "Compute/SpComputeEngine.hpp"
#include "Compute/SpWorkerTeamBuilder.hpp"
#include "TaskGraph/SpTaskGraph.hpp"
#include "Config/SpConfig.hpp"


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

                printf("CPU -- interaction between source %e %e %e and target %e %e %e\n",
                       values[X], values[Y], values[Z],
                       tx, ty, tz);

                printf("CPU -- source dx %e dy %e dz %e pot %e\n", values[FX][idxSource],
                       values[FY][idxSource], values[FZ][idxSource], values[POTENTIAL][idxSource]);
            }

            values[FX][idxTarget] += tfx;
            values[FY][idxTarget] += tfy;
            values[FZ][idxTarget] += tfz;
            values[POTENTIAL][idxTarget] += tpo;

            printf("CPU -- target dx %e dy %e dz %e pot %e\n", values[FX][idxTarget],
                   values[FY][idxTarget], values[FZ][idxTarget], values[POTENTIAL][idxTarget]);
        }
    }

    void compute(const ParticlesGroup& inOther){
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

//            inOther.values[FX][idxTarget] += tfx;
//            inOther.values[FY][idxTarget] += tfy;
//            inOther.values[FZ][idxTarget] += tfz;
//            inOther.values[POTENTIAL][idxTarget] += tpo;
        }
    }

    std::size_t memmovNeededSize() const{
        return sizeof(double)*nbParticles*NB_VALUE_TYPES;
    }

    template <class DeviceMemmov>
    void memmovHostToDevice(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == sizeof(double)*nbParticles*NB_VALUE_TYPES);
        double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            mover.copyHostToDevice(&doubleDevicePtr[idxValueType*nbParticles], values[idxValueType].data(), nbParticles*sizeof(double));
        }
    }

    template <class DeviceMemmov>
    void memmovDeviceToHost(DeviceMemmov& mover, void* devicePtr, std::size_t size){
        assert(size == sizeof(double)*nbParticles*NB_VALUE_TYPES);
        double* doubleDevicePtr = reinterpret_cast<double*>(devicePtr);
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            mover.copyDeviceToHost(values[idxValueType].data(), &doubleDevicePtr[idxValueType*nbParticles], nbParticles*sizeof(double));
        }
    }

    struct View{
        View(){}
        View(void* devicePtr, std::size_t size){}
        View(const void* devicePtr, std::size_t size){}
    };
    using DeviceDataType = View;
};

#ifdef SPECX_COMPILE_WITH_CUDA
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

    for(std::size_t idxTarget = uniqueId ; idxTarget < nbParticles+gridDim.x-1 ; idxTarget += nbThreads){
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

                        printf("GPU -- interaction between source %e %e %e and target %e %e %e\n",
                               sourcesX[otherIndex], sourcesY[otherIndex], sourcesZ[otherIndex],
                               tx, ty, tz);
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

            printf("GPU -- target dx %e dy %e dz %e pot %e\n", values[ParticlesGroup::FX][idxTarget],
                   values[ParticlesGroup::FY][idxTarget], values[ParticlesGroup::FZ][idxTarget], values[ParticlesGroup::POTENTIAL][idxTarget]);
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

    for(std::size_t idxTarget = uniqueId ; idxTarget < nbParticlesTgt+gridDim.x-1 ; idxTarget += nbThreads){
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

#include <iostream>

int main(){
#ifdef SPECX_COMPILE_WITH_CUDA
    SpCudaUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(1));
#endif
    SpTaskGraph tg;

    tg.computeOn(ce);

    const std::size_t NbParticles = 2;
    ParticlesGroup particles(NbParticles);
    FillRandomValues(particles, 0);
    ParticlesGroup particlesB(NbParticles);
    FillRandomValues(particlesB, 1);

    ParticlesGroup cu_particles(particles);
    ParticlesGroup cu_particlesB(particlesB);

    particles.computeSelf();
    particlesB.computeSelf();

    //particles.compute(particlesB);

    tg.task(SpWrite(cu_particles),
//            SpCpu([](ParticlesGroup& particlesW) {
//                particlesW.computeSelf();
//    })
        #ifdef SPECX_COMPILE_WITH_CUDA
            /*,*/ SpCuda([](SpDeviceDataView<ParticlesGroup> paramA) {
                p2p_inner_gpu<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.getRawPtr(), paramA.getRawSize());
            })
        #endif
    );

    tg.task(SpWrite(cu_particlesB),
            SpCpu([](ParticlesGroup& particlesW) {
                particlesW.computeSelf();
    })
//        #ifdef SPECX_COMPILE_WITH_CUDA
//            , SpCuda([](SpDeviceDataView<ParticlesGroup> paramA) {
//                p2p_inner_gpu<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.getRawPtr(), paramA.getRawSize());
//            })
//        #endif
    );

//    tg.task(SpWrite(cu_particles),SpRead(cu_particlesB),
//            SpCpu([](ParticlesGroup& particlesW, const ParticlesGroup& particlesR) {
//                particlesW.compute(particlesR);
//    })
//        #ifdef SPECX_COMPILE_WITH_CUDA
//            , SpCuda([](SpDeviceDataView<ParticlesGroup> paramA, SpDeviceDataView<const ParticlesGroup> paramB) {
//                p2p_neigh_gpu<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.getRawPtr(), paramA.getRawSize(),
//                                                                         paramB.getRawPtr(), paramB.getRawSize());
//            })
//        #endif
//    );

    tg.task(SpWrite(cu_particles),
            SpCpu([](ParticlesGroup& particlesW) {
                particlesW.computeSelf();
            })
    );

    tg.waitAllTasks();

    std::cout << "Error particles = " << ChechAccuracy(particles, cu_particles) << std::endl;
    std::cout << "Error particles = " << ChechAccuracy(particlesB, cu_particlesB) << std::endl;

    return 0;
}

