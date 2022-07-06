###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(LIB_SPETABARU CXX)

ADD_DEFINITIONS(${SPETABARU_CXX_FLAGS})

# Adding the entire project dir as an include dir
INCLUDE_DIRECTORIES(
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_BINARY_DIR}
)

INSTALL(DIRECTORY ./ DESTINATION include/ )

if(SPETABARU_DO_INSTALL)
    INSTALL( FILES "${SPETABARU_BINARY_DIR}/Src/SPETABARUConfig.h" DESTINATION include/Src )
endif()


# Searching all cpp file
file(GLOB_RECURSE source_lib_files ./*.cpp)

if(NOT SPETABARU_COMPILE_WITH_CUDA)
    list(FILTER source_lib_files EXCLUDE REGEX "/Cuda/")
endif()

if(NOT SPETABARU_COMPILE_WITH_MPI)
    list(FILTER source_lib_files EXCLUDE REGEX "/MPI/")
endif()


# Adding cpp files to project
add_library(
	spetabaru
	${source_lib_files}
)

if(SPETABARU_DO_INSTALL)
    target_include_directories(spetabaru INTERFACE $<INSTALL_INTERFACE:include>)
    INSTALL(TARGETS spetabaru EXPORT spetabaru DESTINATION lib )
    INSTALL(EXPORT spetabaru DESTINATION lib )
endif()

