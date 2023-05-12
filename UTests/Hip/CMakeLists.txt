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
	./*.cu
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
		    

    # Adding the project sources dir as an include dir
    target_include_directories(
         ${execname}
         PRIVATE
	     ${SPECX_BINARY_DIR}/Src    
	     ${SPECX_SOURCE_DIR}/Src
	     ${SPECX_SOURCE_DIR}/UTests    
	     ${UTESTS_HIP_SPECX_SOURCE_DIR}
    )

	target_link_libraries(
		${execname}
		specx
		${SPECX_LIBRARIES}
		)
	add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
endforeach(exec)
