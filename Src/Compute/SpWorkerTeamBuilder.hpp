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
#ifdef SPECX_COMPILE_WITH_CUDA
static small_vector<std::unique_ptr<SpWorker>> TeamOfCudaWorkers(const int nbWorkerPerCudas = SpCudaUtils::GetDefaultNbStreams(),
                                             const int nbCudaWorkers = SpCudaUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkerPerCudas*nbCudaWorkers);

    for(int idxCuda = 0; idxCuda < nbCudaWorkers; ++idxCuda) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerCudas; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CUDA_WORKER));
            res.back()->cudaData.init(idxCuda);
        }
    }

    return res;
}

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuCudaWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerCudas = SpCudaUtils::GetDefaultNbStreams(),
                                             const int nbCudaWorkers = SpCudaUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbCpuWorkers + nbWorkerPerCudas*nbCudaWorkers);

    for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CPU_WORKER));
    }

    for(int idxCuda = 0; idxCuda < nbCudaWorkers; ++idxCuda) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerCudas; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CUDA_WORKER));
            res.back()->cudaData.init(idxCuda);
        }
    }

    return res;
}
#endif
static auto DefaultTeamOfWorkers() {
#ifdef SPECX_COMPILE_WITH_CUDA
    return TeamOfCpuCudaWorkers();
#else
    return TeamOfCpuWorkers();
#endif
}

};


#endif // SPWORKERTEAMBUILDER_HPP
