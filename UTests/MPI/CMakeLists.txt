###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
#project(UTESTS_MPI_SPECX LANGUAGES CXX CUDA)
project(UTESTS_MPI_SPECX LANGUAGES CXX)

# Find all code files (no GLOB_RECURSE)
file(
	GLOB	
	source_tests_files
	./*.cpp
	./*.cu
	)


# Adding the project sources dir as an include dir
INCLUDE_DIRECTORIES(
	 ${SPECX_BINARY_DIR}/Src    
	 ${SPECX_SOURCE_DIR}/Src
	 ${SPECX_SOURCE_DIR}/UTests    
	 ${UTESTS_MPI_SPECX_SOURCE_DIR}
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

	add_executable(
			${execname}
			${exec}
		)

	target_link_libraries(
		${execname}
		specx
		${SPECX_LIBRARIES}
		)
	# add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
	add_test(${execname} mpirun -np 2 ${CMAKE_CURRENT_BINARY_DIR}/${execname})
endforeach(exec)
