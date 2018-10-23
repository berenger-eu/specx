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

file(
    GLOB_RECURSE 	 	 	 	
 	hpp_in_dir
	./*.hpp ./*.h
)
FOREACH(hpp_file ${hpp_in_dir})	
    file(RELATIVE_PATH relative_hpp_file ${CMAKE_CURRENT_SOURCE_DIR} ${hpp_file})
    if(SPETABARU_DO_INSTALL)
        INSTALL( FILES ${hpp_file} DESTINATION include/${relative_hpp_file} )
    endif()

    if($ENV{VERBOSE})
        MESSAGE(STATUS "Src -- Add ${relative_hpp_file}")
    endif()
ENDFOREACH()

if(SPETABARU_DO_INSTALL)
    INSTALL( FILES "${SPETABARU_BINARY_DIR}/Src/SPETABARUConfig.h" DESTINATION include/Src )
endif()


# Searching all cpp file
file(
	GLOB_RECURSE
	source_lib_files
	./*.cpp
)

# Adding cpp files to project
add_library(
	spetabaru
	${source_lib_files}
)

