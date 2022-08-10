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

class int_data_holder : public SpAbstractSerializable {
public:
	int_data_holder(int value = 0) : value{value} {}
	int_data_holder(SpDeserializer &deserializer) : value(deserializer.restore<decltype(value)>("value")) {
	}
	
	void serialize(SpSerializer &serializer) const final {
		serializer.append(value, "value");
	}

	int get() const { return value; }
	void set(int value) {
		this->value = value;
	}
private:
	int value;
};

class SimpleUserDefinedSerializationMpiTest : public UTester< SimpleUserDefinedSerializationMpiTest > {
    using Parent = UTester< SimpleUserDefinedSerializationMpiTest >;

    void Test(){
        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers(2));
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        int_data_holder a = 1;
        int_data_holder b = 0;

        tg.computeOn(ce);

        // This test works only with at least 2 processes
        assert(SpMpiUtils::GetMpiSize() >= 2);
        if(SpMpiUtils::GetMpiRank() == 0){
            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.set(paramA.get() + paramB.get());
                        })
            );

            tg.mpiSend(b, 1, 0);
            tg.mpiRecv(b, 1, 1);
        }
        else if(SpMpiUtils::GetMpiRank() == 1){
            tg.mpiRecv(b, 0, 0);

            tg.task(SpRead(a), SpWrite(b),
                        SpCpu([](const auto& paramA, auto& paramB) {
                            paramB.set(paramA.get() + paramB.get());
                        })
            );

            tg.mpiSend(b, 0, 1);
        }

        tg.waitAllTasks();

        UASSERTETRUE(b.get() == 2);
    }


    void SetTests() {
        Parent::AddTest(&SimpleUserDefinedSerializationMpiTest::Test, "Simple User Defined Serialization MPI Test");
    }
};

// You must do this
TestClass(SimpleUserDefinedSerializationMpiTest)
