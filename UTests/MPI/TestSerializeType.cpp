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

struct RawStruct{
    int k;
    int tab[120];
};


class SerializableClass : public SpAbstractSerializable {
public:
    SerializableClass() = default;
    SerializableClass([[maybe_unused]] SpDeserializer& deserializer) {
    }

    void serialize([[maybe_unused]]  SpSerializer& serializer) const final {
    }
};


class DirectAccessClass {
    void * data;
public:
    const unsigned char* getRawData() const {
        return nullptr;
    }
    std::size_t getRawDataSize() const {
        return 0;
    }

    void restoreRawData([[maybe_unused]] const unsigned char* ptr,[[maybe_unused]]  std::size_t size){
    }
};

class TestSerializeType : public UTester< TestSerializeType > {
    using Parent = UTester< TestSerializeType >;

    void Test(){
        static_assert (SpSerializationType::SP_RAW_TYPE == SpGetSerializationType<int>(), "Must be true");
        static_assert (SpSerializationType::SP_RAW_TYPE == SpGetSerializationType<RawStruct>(), "Must be true");
        static_assert (SpSerializationType::SP_RAW_TYPE == SpGetSerializationType<const int&>(), "Must be true");
        static_assert (SpSerializationType::SP_RAW_TYPE == SpGetSerializationType<const RawStruct&>(), "Must be true");

        static_assert (SpSerializationType::SP_VEC_RAW_TYPE == SpGetSerializationType<std::vector<int>>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_RAW_TYPE == SpGetSerializationType<std::vector<RawStruct>>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_RAW_TYPE == SpGetSerializationType<std::vector<int>&>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_RAW_TYPE == SpGetSerializationType<std::vector<RawStruct>&>(), "Must be true");

        static_assert (SpSerializationType::SP_SERIALIZER_TYPE == SpGetSerializationType<SerializableClass>(), "Must be true");
        static_assert (SpSerializationType::SP_SERIALIZER_TYPE == SpGetSerializationType<SerializableClass&>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_SERIALIZER_TYPE == SpGetSerializationType<std::vector<SerializableClass>>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_SERIALIZER_TYPE == SpGetSerializationType<std::vector<SerializableClass>&>(), "Must be true");

        static_assert (SpSerializationType::SP_DIRECT_ACCESS == SpGetSerializationType<DirectAccessClass>(), "Must be true");
        static_assert (SpSerializationType::SP_DIRECT_ACCESS == SpGetSerializationType<DirectAccessClass&>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_DIRECT_ACCESS == SpGetSerializationType<std::vector<DirectAccessClass>>(), "Must be true");
        static_assert (SpSerializationType::SP_VEC_DIRECT_ACCESS == SpGetSerializationType<std::vector<DirectAccessClass>&>(), "Must be true");
    }


    void SetTests() {
        Parent::AddTest(&TestSerializeType::Test, "Basic serialize test");
    }
};

// You must do this
TestClass(TestSerializeType)
