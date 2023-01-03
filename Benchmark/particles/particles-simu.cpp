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
    std::size_t nbParticles;

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

    std::vector<double> values[NB_VALUE_TYPES];

public:
    explicit ParticlesGroup(const std::size_t inNbParticles = 0)
        : nbParticles(inNbParticles){
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[idxValueType].resize(nbParticles, 0);
        }
    }

    void setParticleValues(const std::size_t inIdxParticle,
                           const std::array<double, NB_VALUE_TYPES>& inValues){
        assert(inIdxParticle < nbParticles);
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[idxValueType][inIdxParticle] = inValues[idxValueType];
        }
    }

    auto getNbParticles() const{
        return nbParticles;
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

            for(std::size_t idxSource = 0  ; idxSource < nbParticles ; idxSource += 1){
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
    };
    using DeviceDataType = View;
};

#ifdef SPECX_COMPILE_WITH_CUDA
__global__ void p2p_gpu(void* data, std::size_t size){
//    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
//        ptr[idx]++;
//    }
}
#endif

void p2p_cpu(int* ptr, int size){

}



int main(){
#ifdef SPECX_COMPILE_WITH_CUDA
    SpCudaUtils::PrintInfo();
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
#else
    SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(1));
#endif
    SpTaskGraph tg;

    tg.computeOn(ce);

    ParticlesGroup particles;
#ifdef SPECX_COMPILE_WITH_CUDA
    tg.task(SpWrite(particles),
            SpCuda([](SpDeviceDataView<ParticlesGroup> paramA) {
                p2p_gpu<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.getRawPtr(), paramA.getRawSize());
            })
    );
#endif
    /*
    tg.task(SpWrite(b),
            SpCuda([](SpDeviceDataView<int> paramB) {
            #ifndef SPECX_EMUL_GPU
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramB.objPtr(), 1);
            #else
                (*paramB.objPtr())++;
            #endif
            })
            );

    tg.task(SpRead(a), SpWrite(b),
            SpCpu([](const int& paramA, int& paramB) {
        paramB = paramA + paramB;
    })
            );

    tg.task(SpWrite(a),
            SpCpu([](int& paramA) {
        paramA++;
    }),
            SpCuda(
                [](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.objPtr(), 1);
            #else
                (*paramA.objPtr())++;
            #endif
            })
            );

    tg.task(SpWrite(a), SpWrite(b),
            SpCpu([](int& paramA, int& paramB) {
        paramA++;
        paramB++;
    })
            );


    static_assert(SpDeviceDataView<MemmovClassExample>::MoveType == SpDeviceDataUtils::DeviceMovableType::MEMMOV,
            "should be memmov");

    MemmovClassExample obj;

    tg.task(SpWrite(obj),
            SpCuda([](SpDeviceDataView<MemmovClassExample> objv) {
            })
            );

    tg.task(SpWrite(a),
            SpCuda([](SpDeviceDataView<std::vector<int>> paramA) {
                inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                paramA.nbElements());

                std::this_thread::sleep_for(std::chrono::seconds(2));
            })
            );
*/
    tg.waitAllTasks();

    return 0;
}

