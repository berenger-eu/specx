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
#include "MPI/SpMpiUtils.hpp"
#include "MPI/SpSerializer.hpp"

class IntDataHolder : public SpAbstractSerializable {
public:
    IntDataHolder(int inKey = 0)
        : key{inKey} {
    }

    IntDataHolder(SpDeserializer &deserializer)
        : key(deserializer.restore<decltype(key)>("value")) {
	}
	
	void serialize(SpSerializer &serializer) const final {
        serializer.append(key, "value");
    }

    int& value(){
        return key;
    }

    const int& value() const{
        return key;
    }
private:
    int key;
};

struct RawStruct{
    int key;

    int& value(){
        return key;
    }

    const int& value() const{
        return key;
    }
};

class DirectAccessClass {
    int key;
public:
    const unsigned char* getRawData() const {
        return reinterpret_cast<const unsigned char*>(&key);
    }
    std::size_t getRawDataSize() const {
        return sizeof(key);
    }

    void restoreRawData(const unsigned char* ptr, std::size_t size){
        assert(sizeof(key) == size);
        key = *reinterpret_cast<const int*>(ptr);
    }

    int& value(){
        return key;
    }

    const int& value() const{
        return key;
    }
};


class SimpleUserDefinedSerializationMpiTest : public UTester< SimpleUserDefinedSerializationMpiTest > {
    using Parent = UTester< SimpleUserDefinedSerializationMpiTest >;

    void Test(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        IntDataHolder a = 1;
        IntDataHolder b = 0;

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(b.value() == 2);
    }


    void TestRawStruct(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        RawStruct a{1};
        RawStruct b{0};

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(b.value() == 2);
    }


    void TestDirectAccess(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        DirectAccessClass a;
        a.value() = 1;
        DirectAccessClass b;
        a.value() = 0;

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.value() = (paramA.value() + paramB.value());
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(b.value() == 2);
    }
    void TestVec(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        std::vector<IntDataHolder> a(2);
        for(auto& ae : a){
            ae.value() = 1;
        }
        std::vector<IntDataHolder> b(2);
        for(auto& be : a){
            be.value() = 0;
        }

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        for(auto& be : b){
            UASSERTETRUE(be.value() == 2);
        }
    }


    void TestRawStructVec(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        std::vector<RawStruct> a(2);
        for(auto& ae : a){
            ae.value() = 1;
        }
        std::vector<RawStruct> b(2);
        for(auto& be : a){
            be.value() = 0;
        }

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        for(auto& be : b){
            UASSERTETRUE(be.value() == 2);
        }
    }


    void TestDirectAccessVec(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        std::vector<DirectAccessClass> a(2);
        for(auto& ae : a){
            ae.value() = 1;
        }
        std::vector<DirectAccessClass> b(2);
        for(auto& be : a){
            be.value() = 0;
        }

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            assert(paramA.size() == 2);
                            assert(paramB.size() == 2);
                            for(std::size_t idx = 0 ; idx < paramA.size() ; ++idx){
                                paramB[idx].value() += (paramA[idx].value() + paramB[idx].value());
                            }
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        for(auto& be : b){
            UASSERTETRUE(be.value() == 2);
        }
    }

    void SetTests() {
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::Test, "Simple User Defined Serialization MPI Test");
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::TestRawStruct, "Simple User Defined Serialization MPI Test");
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::TestDirectAccess, "Simple User Defined Serialization MPI Test");
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::TestVec, "Simple User Defined Serialization MPI Test");
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::TestRawStructVec, "Simple User Defined Serialization MPI Test");
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::TestDirectAccessVec, "Simple User Defined Serialization MPI Test");
    }
};

// You must do this
TestClass(SimpleUserDefinedSerializationMpiTest)
