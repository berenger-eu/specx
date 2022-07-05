#ifndef SPMPITYPEUTILS_HPP
#define SPMPITYPEUTILS_HPP

#include <mpi.h>

template <class ObjectType>
struct SpGetMpiType;

template <>
struct SpGetMpiType<char>{
   static constexpr MPI_Datatype type = MPI_CHAR;
};
template <>
struct SpGetMpiType<unsigned char>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};
template <>
struct SpGetMpiType<short>{
   static constexpr MPI_Datatype type = MPI_SHORT;
};
template <>
struct SpGetMpiType<unsigned short>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_SHORT;
};
template <>
struct SpGetMpiType<int>{
   static constexpr MPI_Datatype type = MPI_INT;
};
template <>
struct SpGetMpiType<unsigned>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED;
};
template <>
struct SpGetMpiType<long>{
   static constexpr MPI_Datatype type = MPI_LONG;
};
template <>
struct SpGetMpiType<unsigned long>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG;
};
template <>
struct SpGetMpiType<long long int>{
   static constexpr MPI_Datatype type = MPI_LONG_LONG_INT;
};
template <>
struct SpGetMpiType<float>{
   static constexpr MPI_Datatype type = MPI_FLOAT;
};
template <>
struct SpGetMpiType<double>{
   static constexpr MPI_Datatype type = MPI_DOUBLE;
};
template <>
struct SpGetMpiType<long double>{
   static constexpr MPI_Datatype type = MPI_LONG_DOUBLE;
};
template <>
struct SpGetMpiType<signed char>{
   static constexpr MPI_Datatype type = MPI_SIGNED_CHAR;
};
template <>
struct SpGetMpiType<unsigned long long>{
   static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG_LONG;
};

#endif // SPMPITYPEUTILS_HPP
