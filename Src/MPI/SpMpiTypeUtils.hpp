#ifndef SPMPITYPEUTILS_HPP
#define SPMPITYPEUTILS_HPP

#include <mpi.h>
#include <type_traits>

/// Return the correct MPI data type for a given native data type
template <class ObjectType>
inline constexpr MPI_Datatype SpGetMpiType(){
    if constexpr(std::is_same<char, ObjectType>::value){
        return MPI_CHAR;
    }
    else if constexpr(std::is_same<unsigned char, ObjectType>::value){
        return MPI_UNSIGNED_CHAR;
    }
    else if constexpr(std::is_same<short, ObjectType>::value){
        return MPI_SHORT;
    }
    else if constexpr(std::is_same<unsigned short, ObjectType>::value){
        return MPI_UNSIGNED_SHORT;
    }
    else if constexpr(std::is_same<int, ObjectType>::value){
        return MPI_INT;
    }
    else if constexpr(std::is_same<unsigned, ObjectType>::value){
        return MPI_UNSIGNED;
    }
    else if constexpr(std::is_same<long, ObjectType>::value){
        return MPI_LONG;
    }
    else if constexpr(std::is_same<unsigned long, ObjectType>::value){
        return MPI_UNSIGNED_LONG;
    }
    else if constexpr(std::is_same<long long int, ObjectType>::value){
        return MPI_LONG_LONG_INT;
    }
    else if constexpr(std::is_same<float, ObjectType>::value){
        return MPI_FLOAT;
    }
    else if constexpr(std::is_same<double, ObjectType>::value){
        return MPI_DOUBLE;
    }
    else if constexpr(std::is_same<long double, ObjectType>::value){
        return MPI_LONG_DOUBLE;
    }
    else if constexpr(std::is_same<signed char, ObjectType>::value){
        return MPI_SIGNED_CHAR;
    }
    else if constexpr(std::is_same<unsigned long long, ObjectType>::value){
        return MPI_UNSIGNED_LONG_LONG;
    }
}

#endif // SPMPITYPEUTILS_HPP
