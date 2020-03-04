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
    uint64_t field;
} cpu_set_t;

static void CPU_ZERO(cpu_set_t *cs) {
    cs->field = 0;
}

static void CPU_SET(int num, cpu_set_t *cs) {
    cs->field |= (1 << num);
}

static int CPU_ISSET(int num, cpu_set_t *cs) {
    return (cs->field & (1 << num));
}

namespace macosspecific {
    
inline int sched_getaffinity_np(pthread_t thread, [[maybe_unused]] size_t cpu_set_size, cpu_set_t *mask) {
    thread_port_t mach_thread = pthread_mach_thread_np(thread);
    thread_affinity_policy_data_t affinity_policy;
    mach_msg_type_number_t policy_info_count = THREAD_AFFINITY_POLICY_COUNT;
    boolean_t b;
    
    kern_return_t retValue = thread_policy_get(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&affinity_policy), &policy_info_count, &b);
    
    if(retValue != KERN_SUCCESS) {
        perror("macosspecific::sched_getaffinity error thread_policy_get");
        return -1;
    }
    
    CPU_ZERO(mask);
    CPU_SET(affinity_policy.affinity_tag, mask);
    
    return 0;
}

inline int sched_setaffinity_np(pthread_t thread, size_t cpu_set_size, cpu_set_t *mask) {
    integer_t cpu_id = 0;
    
    /* find core number which is set in cpu_set */
    for (size_t idx = 0; idx < cpu_set_size*CHAR_BIT-1; idx++) {
        if (CPU_ISSET(idx, mask)) {
            cpu_id = idx;
            break;
        }
    }
    
    thread_affinity_policy_data_t affinity_policy = { cpu_id };
    thread_port_t mach_thread = pthread_mach_thread_np(thread);
    
    kern_return_t retValue = thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, reinterpret_cast<thread_policy_t>(&affinity_policy), THREAD_AFFINITY_POLICY_COUNT);
    
    if(retValue != KERN_SUCCESS) {
        perror("macosspecific::sched_setaffinity error thread_policy_set");
        return -1;
    }
    
    return 0;
}

} // end namespace macosspecific

#endif
