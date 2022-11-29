///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#include <utility>
#include <thread>
#include <chrono>

#include "UTester.hpp"

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
        : nbParticles(inNbPartcies){
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[idxValueType].resize(nbParticles, 0);
        }
    }

    void setParticleValues(const std::size_t inIdxParticle,
                           const std::array<double, NB_VALUE_TYPES>& inValues){
        assert(inIdxParticle < nbParticles);
        for(std::size_t idxValueType = 0 ; idxValueType < NB_VALUE_TYPES ; ++idxValueType){
            values[inIdxParticle] = inValues[inIdxParticle];
        }
    }

    void compute(ParticlesGroup& inOther){
        for(std::size_t idxTarget = 0 ; idxTarget < inOther.getNbParticles() ; ++idxTarget){
            const double tx = double(inOther.values[X][idxTarget]);
            const double ty = double(inOther.values[Y][idxTarget]);
            const double tz = double(inOther.values[Z][idxTarget]);
            const double tv = double(inOther.values[PHYSICAL][idxTarget]);

            for(std::size_t idxSource = 0 ; idxSource < nbParticles ; ++idxSource){
                double  tfx = double(0.);
                double  tfy = double(0.);
                double  tfz = double(0.);
                double  tpo = double(0.);

                for( ; idxSource < nbParticlesSources ; idxSource += 1){
                    double dx = tx - double(values[X][idxSource]);
                    double dy = ty - double(values[Y][idxSource]);
                    double dz = tz - double(values[Z][idxSource]);

                    double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                    const double inv_distance = FMath::Sqrt(inv_square_distance);

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
    }

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


__global__ void p2p_gpu(int* ptr, int size){
    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
        ptr[idx]++;
    }
}

void p2p_cpu(int* ptr, int size){
    for(int idx = blockIdx.x*blockDim.x + threadIdx.x ; idx < size ; idx += blockDim.x*gridDim.x){
        ptr[idx]++;
    }
}



class SimpleGpuTest : public UTester< SimpleGpuTest > {
    using Parent = UTester< SimpleGpuTest >;

    void Test(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        int a = 0;
        int b = 0;

        tg.computeOn(ce);

        tg.task(SpWrite(a),
                SpCuda([](SpDeviceDataView<int> paramA) {
            #ifndef SPECX_EMUL_GPU
                    inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.objPtr(), 1);
            #else
                    (*paramA.objPtr())++;
            #endif
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                })
                );

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

        tg.waitAllTasks();

        UASSERTETRUE(a == 3);
        UASSERTETRUE(b == 3);
    }

    void TestVec(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        std::vector<int> a(100,0);
        std::vector<int> b(100,0);

        static_assert(SpDeviceDataView<std::vector<int>>::MoveType == SpDeviceDataUtils::DeviceMovableType::STDVEC,
                "should be stdvec");

        tg.computeOn(ce);

        tg.task(SpWrite(a),
                SpCuda([](SpDeviceDataView<std::vector<int>> paramA) {
                    inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                    paramA.nbElements());

                    std::this_thread::sleep_for(std::chrono::seconds(2));
                })
                );

        tg.task(SpWrite(b),
                SpCuda([](SpDeviceDataView<std::vector<int>> paramB) {
                    inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramB.array(),
                    paramB.nbElements());
                })
                );

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
                SpCuda([](SpDeviceDataView<std::vector<int>> paramA) {
            inc_var<<<1,1,0,SpCudaUtils::GetCurrentStream()>>>(paramA.array(),
                                                               paramA.nbElements());
        })
        );

        tg.task(SpWrite(a), SpWrite(b),
                SpCpu([](std::vector<int>& paramA, std::vector<int>& paramB) {
                    for(auto& va : paramA){
                        va++;
                    }
                    for(auto& vb : paramB){
                        vb++;
                    }
                })
                );

        tg.waitAllTasks();

        for(auto& va : a){
            UASSERTETRUE(va == 3);
        }
        for(auto& vb : b){
            UASSERTETRUE(vb == 3);
        }
    }


    void TestMemMove(){
        SpCudaUtils::PrintInfo();

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuCudaWorkers(1,1,2));
        SpTaskGraph tg;
        tg.computeOn(ce);

        static_assert(SpDeviceDataView<MemmovClassExample>::MoveType == SpDeviceDataUtils::DeviceMovableType::MEMMOV,
                "should be memmov");

        MemmovClassExample obj;

        tg.task(SpWrite(obj),
                SpCuda([](SpDeviceDataView<MemmovClassExample> objv) {
                })
                );

        tg.waitAllTasks();
    }

    void SetTests() {
        Parent::AddTest(&SimpleGpuTest::Test, "Basic gpu test");
        Parent::AddTest(&SimpleGpuTest::TestVec, "Basic gpu test with vec");
        Parent::AddTest(&SimpleGpuTest::TestMemMove, "Basic gpu test with memmov");
    }
};

// You must do this
TestClass(SimpleGpuTest)
