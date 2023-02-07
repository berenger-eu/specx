#ifndef UTESTERMPI_HPP
#define UTESTERMPI_HPP

#include "UTester.hpp"
#include "Command.hpp"
#include "MPI/SpMpiUtils.hpp"

#include <iostream>

#define TestClassMpi(X, NP)\
    int main(int argc, char ** argv){\
        if(SpMpiUtils::GetMpiSize() == 1){\
            std::string fullCommand = "bash ../UTests/MPI/mpi_script.sh " + std::to_string(NP) + " " + argv[0];\
            std::cout << fullCommand << std::endl;\
            Command cmd;\
            cmd.Command = fullCommand;\
            cmd.StdIn = "";\
            cmd.execute();\
            std::cout << cmd.StdOut;\
            std::cerr << cmd.StdErr;\
            return cmd.ExitStatus;\
        }\
        else{\
            X Controller;\
            return Controller.Run();\
        }\
    }

#endif // UTESTERMPI_HPP
