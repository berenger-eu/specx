#ifndef SPMPIUTILS_HPP
#define SPMPIUTILS_HPP

#include "Config/SpConfig.hpp"

#ifndef SPECX_COMPILE_WITH_MPI
#error MPI but be enable to use this file.
#endif

#include <mpi.h>
#include <iostream>
#include <vector>

/// Test if an MPI is OK, else exit
#define SpAssertMpi(X) if(MPI_SUCCESS != (X)) { std::cerr << "MPI Error at line " << __LINE__ << std::endl; std::cerr.flush() ; exit(1); }

///
/// \brief The SpMpiUtils class is an interface to MPI
/// functions.
///
class SpMpiUtils{
public:

static int GetMpiSize(MPI_Comm comm = MPI_COMM_WORLD){
    int comm_size;
    SpAssertMpi(MPI_Comm_size( comm, &comm_size ));
    return comm_size;
}

static int GetMpiRank(MPI_Comm comm = MPI_COMM_WORLD){
    int my_rank;
    SpAssertMpi(MPI_Comm_rank( comm, &my_rank ));
    return my_rank;
}

template <class ItemType, class ... Args>
static std::vector<ItemType> BuildVec(Args&& ... args){
    std::vector<ItemType> vec;
    (vec.push_back(std::forward<Args>(args)), ...);
    return vec;
}


};
#endif // SPMPIUTILS_HPP
