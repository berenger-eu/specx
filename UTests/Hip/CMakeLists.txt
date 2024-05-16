###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(UTESTS_HIP_SPECX LANGUAGES CXX)

# Find all code files (no GLOB_RECURSE)
file(
	GLOB	
	source_tests_files
	./*.cpp
	)

if(SPECX_COMPILE_WITH_HIP)
    set_source_files_properties(axpy_hip.cpp PROPERTIES LANGUAGE ${GPU_RUNTIME})
    include_directories(/opt/rocm-5.7.2/hip/include)
endif()



# Adding the project sources dir as an include dir
INCLUDE_DIRECTORIES(
	 ${SPECX_BINARY_DIR}/Src    
	 ${SPECX_SOURCE_DIR}/Src
	 ${SPECX_SOURCE_DIR}/UTests    
	 ${UTESTS_HIP_SPECX_SOURCE_DIR}
)


# Add execs - 1 cpp = 1 exec
foreach(exec ${source_tests_files}) 
	get_filename_component(
		execname ${exec}
		NAME_WE
	)
	if($ENV{VERBOSE})
		MESSAGE(STATUS "UTests -- Add ${execname}")
	endif()


	if(SPECX_COMPILE_WITH_CUDA)
		set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
	endif()

	if(SPECX_COMPILE_WITH_HIP)
		set_source_files_properties(${exec} PROPERTIES LANGUAGE HIP)
	endif()


	add_executable(
			${execname}
			${exec}
		)

	target_link_libraries(
		${execname}
		specx
		${SPECX_LIBRARIES}
		)
	add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
endforeach(exec)
