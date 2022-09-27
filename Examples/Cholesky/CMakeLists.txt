###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(CHOLESKY_SPECX CXX)

ADD_DEFINITIONS(${SPECX_CXX_FLAGS})


if(SPECX_BUILD_EXAMPLES)
    OPTION( SPECX_BUILD_CHOLESKY_EXAMPLES  "Set to ON to build cholesky example" ON )

    if(SPECX_BUILD_CHOLESKY_EXAMPLES)
        if($ENV{VERBOSE})
            MESSAGE(STATUS "Examples -- SPECX_CXX_FLAGS   : ${SPECX_CXX_FLAGS}")
        endif()

        # Find all code files
        file(	
	        GLOB_RECURSE
	        source_tests_files
	        ./*.cpp
	        )


        # Adding the project sources dir as an include dir
        INCLUDE_DIRECTORIES(
             ${SPECX_BINARY_DIR}/Src    
             ${SPECX_SOURCE_DIR}/Src
        )

        LINK_DIRECTORIES(
        )

        # Add execs - 1 cpp = 1 exec
        foreach(exec ${source_tests_files}) 
	        get_filename_component(
		        execname ${exec}
		        NAME_WE
	        )

            if($ENV{VERBOSE})
                MESSAGE(STATUS "Patterns -- Add ${execname}")
            endif()

            add_executable(
			        ${execname}
			        ${exec}
		        )

            set_target_properties(
                ${execname}
                PROPERTIES COMPILE_FLAGS "-llapack -lblas"
            )
            target_link_libraries(
                ${execname}
                specx
	            ${SPECX_LIBRARIES}
                "-llapack -lblas"
                ) 
        endforeach(exec)
    endif()
endif()
