///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDEBUG_HPP
#define SPDEBUG_HPP

#include <mutex>
#include <sstream>
#include <iostream>
#include <cstring>

#include "Config/SpConfig.hpp"

#ifdef SPECX_COMPILE_WITH_MPI
#include "MPI/SpMpiUtils.hpp"
#endif

/**
 * This class should be used to print debug info.
 * The environment variable SPECX_USE_DEBUG_PRINT allow to disable/enable
 * the output at runtime.
 */
class SpDebug {
#ifdef SPECX_USE_DEBUG_PRINT
    const bool hasBeenEnabled;
    std::mutex outputMutex;
    const bool toFile;

    SpDebug()
        : hasBeenEnabled(getenv("SPECX_DEBUG_PRINT") && strcmp(getenv("SPECX_DEBUG_PRINT"),"TRUE") == 0?true:false),
          toFile(false){
    }
#else
    SpDebug(){}
#endif

    SpDebug(const SpDebug&) = delete;
    SpDebug(SpDebug&&) = delete;
    SpDebug& operator=(const SpDebug&) = delete;
    SpDebug& operator=(SpDebug&&) = delete;

public:
    static SpDebug Controller;

    class Printer {
        SpDebug& master;

#ifdef SPECX_USE_DEBUG_PRINT
        std::stringstream buffer;
#endif
        explicit Printer(SpDebug& inMaster) : master(inMaster){
#ifdef SPECX_USE_DEBUG_PRINT
            if(master.isEnable()){                
#ifdef SPECX_COMPILE_WITH_MPI
                buffer << "[MPI-" << SpMpiUtils::GetMpiRank() << "] ";
#endif
                buffer << "[THREAD-" << master.getThreadId() << "] ";
            }
#endif
        }

    public:
        ~Printer(){            
#ifdef SPECX_USE_DEBUG_PRINT
            if(master.isEnable()){
                buffer << "\n";
                const std::string toOutput = buffer.str();
                if(master.toFile){

                }
                else{
                    master.outputMutex.lock();
                    std::cout << toOutput;
                    master.outputMutex.unlock();
                }
            }
#endif
        }

        Printer(const Printer&) = delete;
        Printer(Printer&&) = default;
        Printer& operator=(const Printer&) = delete;
        Printer& operator=(Printer&&) = delete;

        template <class Param>
        Printer& operator<<([[maybe_unused]] Param&& toOutput){
#ifdef SPECX_USE_DEBUG_PRINT
            if(master.isEnable()){
                buffer << toOutput;
            }
#endif
            return *this;
        }

        void lineBreak(){
#ifdef SPECX_USE_DEBUG_PRINT
            if(master.isEnable()){
                buffer << '\n';
#ifdef SPECX_COMPILE_WITH_MPI
                buffer << "[MPI-" << SpMpiUtils::GetMpiRank() << "] ";
#endif
                buffer << "[THREAD-" << master.getThreadId() << "] ";
            }
#endif
        }

        friend SpDebug;
    };

    Printer getPrinter(){
        return Printer(*this);
    }

#ifdef SPECX_USE_DEBUG_PRINT
    bool isEnable() const{
        return hasBeenEnabled;
    }
#else
    constexpr bool isEnable() const{
        return false;
    }
#endif

    long int getThreadId() const;

    friend Printer;
};


inline SpDebug::Printer SpDebugPrint(){
    return SpDebug::Controller.getPrinter();
}


#endif
