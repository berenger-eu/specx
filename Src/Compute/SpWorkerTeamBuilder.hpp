#ifndef SPWORKERTEAMBUILDER_HPP
#define SPWORKERTEAMBUILDER_HPP

#include "Config/SpConfig.hpp"

#include "SpWorker.hpp"

class SpWorkerTeamBuilder {
public:

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuWorkers(const int nbWorkers = SpUtils::DefaultNumThreads()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkers);

    for(int idxWorker = 0; idxWorker < nbWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CPU_WORKER));
    }

    return res;
}
#ifdef SPETABARU_COMPILE_WITH_CUDA
static small_vector<std::unique_ptr<SpWorker>> TeamOfGpuWorkers(const int nbWorkerPerGpus = SpCudaUtils::GetDefaultNbStreams(),
                                             const int nbGpuWorkers = SpCudaUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkerPerGpus*nbGpuWorkers);

    for(int idxGpu = 0; idxGpu < nbGpuWorkers; ++idxGpu) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerGpus; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::GPU_WORKER));
            res.back()->gpuData.init(idxGpu);
        }
    }

    return res;
}

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuGpuWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerGpus = SpCudaUtils::GetDefaultNbStreams(),
                                             const int nbGpuWorkers = SpCudaUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbCpuWorkers + nbWorkerPerGpus*nbGpuWorkers);

    for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CPU_WORKER));
    }

    for(int idxGpu = 0; idxGpu < nbGpuWorkers; ++idxGpu) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerGpus; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::GPU_WORKER));
            res.back()->gpuData.init(idxGpu);
        }
    }

    return res;
}
#endif
static auto DefaultTeamOfWorkers() {
#ifdef SPETABARU_COMPILE_WITH_CUDA
    return TeamOfCpuGpuWorkers();
#else
    return createTeamOfCpuWorkers();
#endif
}

};


#endif // SPWORKERTEAMBUILDER_HPP
