#ifndef SPMPIUTILS_HPP
#define SPMPIUTILS_HPP

#include "Config/SpConfig.hpp"

#ifndef SPETABARU_COMPILE_WITH_MPI
#error MPI but be enable to use this file.
#endif

#include <mpi.h>
#include <iostream>
#include <vector>

#define SpAssertMpi(X) if(MPI_SUCCESS != (X)) { std::cerr << "MPI Error at line " << __LINE__ << std::endl; std::cerr.flush() ; throw std::runtime_error("Stop from from mpi error"); }

int DpGetMpiSize(MPI_Comm comm = MPI_COMM_WORLD){
    int comm_size;
    SpAssertMpi(MPI_Comm_size( comm, &comm_size ));
    return comm_size;
}

int DpGetMpiRank(MPI_Comm comm = MPI_COMM_WORLD){
    int my_rank;
    SpAssertMpi(MPI_Comm_rank( comm, &my_rank ));
    return my_rank;
}

template <class ItemType, class ... Args>
std::vector<ItemType> DpBuildVec(Args&& ... args){
    std::vector<ItemType> vec;
    (vec.push_back(std::forward<Args>(args)), ...);
    return vec;
}

template <class ObjectType>
struct DpGetMpiType;


template <>
struct DpGetMpiType<char>{
   static constexpr MPI_Datatype type = MPI_CHAR;
};
template <>
struct DpGetMpiType<unsigned char>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};
template <>
struct DpGetMpiType<short>{
   static constexpr MPI_Datatype type = MPI_SHORT;
};
template <>
struct DpGetMpiType<unsigned short>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_SHORT;
};
template <>
struct DpGetMpiType<int>{
   static constexpr MPI_Datatype type = MPI_INT;
};
template <>
struct DpGetMpiType<unsigned>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED;
};
template <>
struct DpGetMpiType<long>{
   static constexpr MPI_Datatype type = MPI_LONG;
};
template <>
struct DpGetMpiType<unsigned long>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG;
};
template <>
struct DpGetMpiType<long long int>{
   static constexpr MPI_Datatype type = MPI_LONG_LONG_INT;
};
template <>
struct DpGetMpiType<float>{
   static constexpr MPI_Datatype type = MPI_FLOAT;
};
template <>
struct DpGetMpiType<double>{
   static constexpr MPI_Datatype type = MPI_DOUBLE;
};
template <>
struct DpGetMpiType<long double>{
   static constexpr MPI_Datatype type = MPI_LONG_DOUBLE;
};
template <>
struct DpGetMpiType<signed char>{
   static constexpr MPI_Datatype type = MPI_SIGNED_CHAR;
};
template <>
struct DpGetMpiType<unsigned long long>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG_LONG;
};
#endif // SPMPIUTILS_HPP
