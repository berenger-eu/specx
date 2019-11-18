///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPUTILS_HPP
#define SPUTILS_HPP

#include <thread>
#include <string>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include <cassert>

#include "SpDebug.hpp"

/**
 * Utils methods
 */
namespace SpUtils{
    /** Return the default number of threads using hardware's capacity */
    inline int DefaultNumThreads(){
        return static_cast<int>(std::thread::hardware_concurrency());
    }

    inline void BindToCore(const int inCoreId){
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(inCoreId, &set);

        pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
        int retValue = sched_setaffinity(tid, sizeof(set), &set);
        assert(retValue == 0);

        SpDebugPrint() << "Bind to " << inCoreId << " core";
    }

    /** Return the current binding (bit a position n is set to 1 when bind
     * to core n)
     */
    inline long int GetBinding(){
        cpu_set_t mask;
        CPU_ZERO(&mask);
        pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
        // Get the affinity
        int retValue = sched_getaffinity(tid, sizeof(mask), &mask);
        assert(retValue == 0);
        long int retMask = 0;
        for(size_t idx = 0 ; idx < sizeof(long int)*8-1 ; ++idx){
            if(CPU_ISSET(idx, &mask)){
                retMask |= (1<<idx);
            }
        }
        return retMask;
    }

    /** Return the current thread id */
    long int GetThreadId();

    /** Return true if the code is executed from a task */
    inline bool IsInTask(){
        return GetThreadId() != 0;
    }

    /** Set current thread Id */
    void SetThreadId(const long int inThreadId);

    /** Replace all substring in a string */
    inline std::string ReplaceAllInString(std::string sentence, const std::string& itemToReplace, const std::string& inSubstitutionString){
        for(std::string::size_type idxFound = sentence.find(itemToReplace); idxFound != std::string::npos; idxFound = sentence.find(itemToReplace)){
            sentence.replace(idxFound, itemToReplace.size(), inSubstitutionString);
        }
        return sentence;
    }

    /** To perform some assert/check */
    inline void CheckCorrect(const char* test, const bool isCorrect, const int line, const char* file){
        if(!isCorrect){
            std::cout << "Error in file " << file << " line " << line << std::endl;
            std::cout << "Test was " << test << std::endl;
            exit(-1);
        }
    }
}

#define spetabaru_xstr(s) spetabaru_str(s)
#define spetabaru_str(s) #s
#define always_assert(X) SpUtils::CheckCorrect(spetabaru_str(X), (X), __LINE__, __FILE__)

#endif
