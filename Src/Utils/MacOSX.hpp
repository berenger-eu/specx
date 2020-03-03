/*
 * Stephane Genaud
 * adapted from https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html
 **/
#ifndef MACOSX_HPP
#define MACOSX_HPP

/**
 *
 *  THE FOLLOWING IS MACOS  SPECIFIC
 *
 **/

#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <mach/kern_return.h>

typedef struct cpu_set {
    uint32_t count;
} cpu_set_t;

static void CPU_ZERO(cpu_set_t *cs) {
    cs->count = 0;
}

static void CPU_SET(int num, cpu_set_t *cs) {
    cs->count |= (1 << num);
}

static int CPU_ISSET(int num, cpu_set_t *cs) {
    return (cs->count & (1 << num));
}

namespace macosspecific {
    
inline int sched_getaffinity(pthread_t thread, size_t cpu_set_size, cpu_set_t *mask) {
    thread_port_t mach_thread = pthread_mach_thread_np(thread);
    thread_affinity_policy_data_t policy;
    
    kern_return_t retValue = thread_policy_get(mach_thread, THREAD_AFFINITY_POLICY, static_cast<thread_policy_t>(&policy), 1);
    
    if(retValue != KERN_SUCCESS) {
        perror("macosspecific::sched_getaffinity error thread_policy_get");
        return -1;
    }
    
    CPU_ZERO(mask);
    CPU_SET(policy.cpu_id, mask);
    
    return 0;
}

inline int sched_setaffinity(pthread_t thread, size_t cpu_set_size, cpu_set_t *mask) {
    int cpu_id = 0;
    
    /* find core number which is set in cpu_set */
    for (size_t idx = 0; idx < cpu_set_size*8-1; idx++) {
        if (CPU_ISSET(idx, mask)) {
            cpu_id = idx;
            break;
        }
    }
    
    thread_affinity_policy_data_t policy = { cpu_id };
    thread_port_t mach_thread = pthread_mach_thread_np(thread);
    
    kern_return_t retValue = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, static_cast<thread_policy_t>(&policy), 1);
    
    if(retValue != KERN_SUCCESS) {
        perror("macosspecific::sched_setaffinity error thread_policy_set");
        return -1;
    }
    
    return 0;
}

} // end namespace macosspecific

#endif
