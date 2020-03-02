/*
 * Stephane Genaud
 * adapted from https://yyshen.github.io/2015/01/18/binding_threads_to_cores_osx.html
 **/


/**
 *
 *  THE FOLLOWING IS MACOS  SPECIFIC
 *
 **/

#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>

#define SYSCTL_CORE_COUNT   "machdep.cpu.core_count"

typedef struct cpu_set {
    uint32_t count;
} cpu_set_t;

static inline void CPU_ZERO(cpu_set_t *cs) {
    cs->count = 0;
}

static inline void CPU_SET(int num, cpu_set_t *cs) {
    cs->count |= (1 << num);
}

static inline int CPU_ISSET(int num, cpu_set_t *cs) {
    return (cs->count & (1 << num));
}

inline int get_cores_num() {
    int core_count = 0;
    size_t  len = sizeof(core_count);
    int ret = sysctlbyname(SYSCTL_CORE_COUNT, &core_count, &len, 0, 0);
    /*  Upon successful completion, sysctlbyname returns 0. Otherwise the value -1 
      is returned and the global variable errno is set to indicate the error. */
    if (ret == -1) {
        perror("* Error: unable to get core count: ");
        return -1;
    }
    return core_count;
}

inline int sched_getaffinity(pid_t pid, size_t cpu_size, cpu_set_t *cpu_set) {
    /* pid_t pid is unused */
    int core_count =  get_cores_num();
    
    if (core_count == -1) {
        return -1;
    }
    
    cpu_set->count = 0;
    
    for (int i = 0; i < core_count; i++) {
        cpu_set->count |= (1 << i);
    }
    return 0;
}

inline int pthread_setaffinity_np(pthread_t thread, size_t cpu_size, cpu_set_t *cpu_set) {
    thread_port_t mach_thread;
    size_t core = 0;   //  thread_affinity_policy_data_t requires it to be a signed int, not a size_t
    int core_count =  get_cores_num();
    
    if (core_count == -1) {
        return -1;
    }

    /* find which CPU is set */
    for (core = 0; core < (core_count * cpu_size); core++) {
        if (CPU_ISSET(core, cpu_set)) {
            break;
        }
    }
    
    int core_assigned = core;
    thread_affinity_policy_data_t policy = { core_assigned };
    
    mach_thread = pthread_mach_thread_np(thread);
    
    thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY, static_cast<thread_policy_t>(&policy), 1);
    
    return 0;
}
