///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDEBUG_HPP
#define SPDEBUG_HPP

#include <mutex>
#include <sstream>
#include <iostream>
#include <cstring>

/**
 * This class should be used to print debug info.
 * The environment variable SPETABARU_DEBUG_PRINT allow to disable/enable
 * the output at runtime.
 */
class SpDebug {
    const bool hasBeenEnabled;
    std::mutex outputMutex;
    const bool toFile;

    SpDebug()
        : hasBeenEnabled(getenv("SPETABARU_DEBUG_PRINT") && strcmp(getenv("SPETABARU_DEBUG_PRINT"),"TRUE") == 0?true:false),
          toFile(false){
    }

    SpDebug(const SpDebug&) = delete;
    SpDebug(SpDebug&&) = delete;
    SpDebug& operator=(const SpDebug&) = delete;
    SpDebug& operator=(SpDebug&&) = delete;

public:
    static SpDebug Controller;

    class Printer {
        SpDebug& master;

        std::stringstream buffer;

        explicit Printer(SpDebug& inMaster) : master(inMaster){
            if(master.isEnable()){
                buffer << "[THREAD-" << master.getThreadId() << "] ";
            }
        }

    public:
        ~Printer(){
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
        }

        Printer(const Printer&) = delete;
        Printer(Printer&&) = default;
        Printer& operator=(const Printer&) = delete;
        Printer& operator=(Printer&&) = delete;

        template <class Param>
        Printer& operator<<(Param&& toOutput){
            if(master.isEnable()){
                buffer << toOutput;
            }
            return *this;
        }

        void lineBreak(){
            if(master.isEnable()){
                buffer << '\n' << "[THREAD-" << master.getThreadId() << "] ";
            }
        }

        friend SpDebug;
    };

    Printer getPrinter(){
        return Printer(*this);
    }

    bool isEnable() const{
        return hasBeenEnabled;
    }

    long int getThreadId() const;

    friend Printer;
};


inline SpDebug::Printer SpDebugPrint(){
    return SpDebug::Controller.getPrinter();
}


#endif
