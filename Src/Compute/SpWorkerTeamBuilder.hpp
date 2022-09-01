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
#ifdef SPECX_COMPILE_WITH_HIP
static small_vector<std::unique_ptr<SpWorker>> TeamOfHipWorkers(const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             const int nbHipWorkers = SpHipUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkerPerHips*nbHipWorkers);

    for(int idxHip = 0; idxHip < nbHipWorkers; ++idxHip) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerHips; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::HIP_WORKER));
            res.back()->hipData.init(idxHip);
        }
    }

    return res;
}

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuHipWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             const int nbHipWorkers = SpHipUtils::GetNbDevices()) {
    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbCpuWorkers + nbWorkerPerHips*nbHipWorkers);

    for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::CPU_WORKER));
    }

    for(int idxHip = 0; idxHip < nbHipWorkers; ++idxHip) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerHips; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorker::SpWorkerType::HIP_WORKER));
            res.back()->hipData.init(idxHip);
        }
    }

    return res;
}
#endif
static auto DefaultTeamOfWorkers() {
#if defined(SPECX_COMPILE_WITH_CUDA)
    return TeamOfCpuCudaWorkers(); 
#elif defined(SPECX_COMPILE_WITH_HIP)
    return TeamOfCpuHipWorkers();
#else
    return TeamOfCpuWorkers();
#endif
}

};


#endif // SPWORKERTEAMBUILDER_HPP
