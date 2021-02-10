///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPUTILS_HPP
#define SPUTILS_HPP

#include <thread>
#include <string>
#include <sstream>
#include <cstring>
#include <functional>
#include <memory>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>

#ifdef __APPLE__
#include "MacOSX.hpp"
#else    
#include <sched.h>
#endif

#include <cassert>

#include "SpDebug.hpp"
#include "small_vector.hpp"

/**
 * Utils methods
 */
namespace SpUtils{
    /** Return the default number of threads using hardware's capacity */
    inline int DefaultNumThreads(){
        if(getenv("OMP_NUM_THREADS")){
            std::istringstream iss(getenv("OMP_NUM_THREADS"),std::istringstream::in);
            int nbThreads = -1;
            iss >> nbThreads;
            if( /*iss.tellg()*/ iss.eof() ) return nbThreads;
        }
        return static_cast<int>(std::thread::hardware_concurrency());
    }

    inline bool DefaultToBind(const int inNumThreads){
        if(getenv("OMP_PROC_BIND")){
            return strcmp(getenv("OMP_PROC_BIND"), "TRUE") == 0
                    || strcmp(getenv("OMP_PROC_BIND"), "true") == 0
                    || strcmp(getenv("OMP_PROC_BIND"), "1") == 0;
        }
        return (inNumThreads <= static_cast<int>(std::thread::hardware_concurrency()));
    }

    inline void BindToCore(const int inCoreId){
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(inCoreId, &mask);
        
        #ifdef __APPLE__
            [[maybe_unused]] int retValue = macosspecific::sched_setaffinity_np(pthread_self(), sizeof(mask), &mask);
            assert(retValue == 0);
        #else
            pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
            [[maybe_unused]] int retValue = sched_setaffinity(tid, sizeof(mask), &mask);
            assert(retValue == 0);
        #endif

        SpDebugPrint() << "Bind to " << inCoreId << " core";
    }

    /** Return the current binding (bit a position n is set to 1 when bind
     * to core n)
     */
    inline void GetBinding(cpu_set_t *mask){
        CPU_ZERO(mask);
        
        #ifdef __APPLE__
            [[maybe_unused]] int retValue = macosspecific::sched_getaffinity_np(pthread_self(), sizeof(*mask), mask);
            assert(retValue == 0);
        #else
            pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
            // Get the affinity
            [[maybe_unused]] int retValue = sched_getaffinity(tid, sizeof(*mask), mask);
            assert(retValue == 0);
        #endif
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

    template <typename CallableTy, typename TupleTy, std::size_t... Is,
    std::enable_if_t<sizeof...(Is) == 0 || std::conjunction_v<std::is_invocable<CallableTy, std::tuple_element_t<Is, std::remove_reference_t<TupleTy>>>...>, int> = 0>
    inline void foreach_in_tuple_impl(CallableTy &&c, TupleTy &&t, std::index_sequence<Is...>) {
        if constexpr(sizeof...(Is) > 0) {
            using RetTy = std::invoke_result_t<CallableTy, std::tuple_element_t<0, std::remove_reference_t<TupleTy>>>;

            if constexpr(std::is_same_v<RetTy, bool>) {
                (std::invoke(std::forward<CallableTy>(c), std::get<Is>(std::forward<TupleTy>(t))) || ...);
            } else {
                (static_cast<void>(std::invoke(std::forward<CallableTy>(c), std::get<Is>(std::forward<TupleTy>(t)))),...);
            }
        }
    }
    
    template <typename CallableTy, typename TupleTy, std::size_t... Is,
    std::enable_if_t<sizeof...(Is) != 0 && std::conjunction_v<std::is_invocable<CallableTy, std::integral_constant<size_t, Is>,
    std::tuple_element_t<Is, std::remove_reference_t<TupleTy>>>...>, int> = 0>
    inline void foreach_in_tuple_impl(CallableTy &&c, TupleTy &&t, std::index_sequence<Is...>) {
        if constexpr(sizeof...(Is) > 0) {
            using RetTy = std::invoke_result_t<CallableTy, std::integral_constant<size_t, 0>, std::tuple_element_t<0, std::remove_reference_t<TupleTy>>>;

            if constexpr(std::is_same_v<RetTy, bool>) {
                (std::invoke(std::forward<CallableTy>(c), std::integral_constant<size_t, Is>{}, std::get<Is>(std::forward<TupleTy>(t))) || ...);
            } else {
                (static_cast<void>(std::invoke(std::forward<CallableTy>(c), std::integral_constant<size_t, Is>{}, std::get<Is>(std::forward<TupleTy>(t)))),...);
            }
        }
    }

    template <typename CallableTy, typename TupleTy>
    inline void foreach_in_tuple(CallableTy &&c, TupleTy &&t) {
        foreach_in_tuple_impl(std::forward<CallableTy>(c), std::forward<TupleTy>(t), std::make_index_sequence<std::tuple_size_v<std::remove_reference_t<TupleTy>>>{});
    }
    
    template <typename Func, std::size_t... Is>
	static void foreach_index(Func&& f, std::index_sequence<Is...>) {
		((void)std::invoke(std::forward<Func>(f), std::integral_constant<Is>{}), ...);
	}

    template <typename T, template<typename T2> class Test, typename=std::void_t<>>
    struct detect : std::false_type {};
    
    template <typename T, template <typename T2> class Test>
    struct detect<T, Test, std::void_t<Test<T>>> : std::true_type {};
    
}

#define spetabaru_xstr(s) spetabaru_str(s)
#define spetabaru_str(s) #s
#define always_assert(X) SpUtils::CheckCorrect(spetabaru_str(X), (X), __LINE__, __FILE__)

#endif
