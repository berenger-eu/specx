###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(LIB_SPECX CXX)

ADD_DEFINITIONS(${SPECX_CXX_FLAGS})

# Adding the entire project dir as an include dir
INCLUDE_DIRECTORIES(
     ${CMAKE_CURRENT_SOURCE_DIR}
     ${CMAKE_CURRENT_BINARY_DIR}
)
include(GNUInstallDirs)
INSTALL(DIRECTORY ./ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/specx )

if(SPECX_DO_INSTALL)
    INSTALL( FILES "${SPECX_BINARY_DIR}/Src/Config/SpConfig.hpp" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/specx/Config )
endif()


# Searching all cpp file
file(GLOB_RECURSE source_lib_files ./*.cpp)

if(NOT SPECX_COMPILE_WITH_CUDA)
    list(FILTER source_lib_files EXCLUDE REGEX "/Cuda/")
endif()

if(NOT SPECX_COMPILE_WITH_MPI)
    list(FILTER source_lib_files EXCLUDE REGEX "/MPI/")
endif()

if(NOT SPECX_COMPILE_WITH_HIP)
    list(FILTER source_lib_files EXCLUDE REGEX "/Hip/")
else()
    foreach(src_file ${source_lib_files}) 
        message(STATUS "${src_file} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1")
        set_source_files_properties(${src_file} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1) 
    endforeach()
endif()


# Adding cpp files to project
add_library(
	specx
	${source_lib_files}
)

if(SPECX_DO_INSTALL)
    target_include_directories(specx INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/specx>)
    INSTALL(TARGETS specx EXPORT specx DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(EXPORT specx DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

