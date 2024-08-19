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
        res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::CPU_WORKER));
    }

    return res;
}
#ifdef SPECX_COMPILE_WITH_CUDA
static small_vector<std::unique_ptr<SpWorker>> TeamOfCudaWorkers(const int nbWorkerPerCudas = SpCudaUtils::GetDefaultNbStreams(),
                                             int nbCudaWorkers = SpCudaUtils::GetNbDevices()) {
    if(SpCudaUtils::GetNbDevices() < nbCudaWorkers){
        std::cout << "[SPECX] The number of devices asked ("
                  << nbCudaWorkers << ") is above the real number of devices ("
                  << SpCudaUtils::GetNbDevices() << ")" << std::endl;
        std::cout << "[SPECX] The real number will be used instead." << std::endl;
        nbCudaWorkers = SpCudaUtils::GetNbDevices();
    }

    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkerPerCudas*nbCudaWorkers);

    for(int idxCuda = 0; idxCuda < nbCudaWorkers; ++idxCuda) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerCudas; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::CUDA_WORKER));
            res.back()->cudaData.init(idxCuda);
        }
    }

    return res;
}

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuCudaWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             int nbCudaWorkers = SpCudaUtils::GetNbDevices(),
                                             const int nbWorkerPerCudas = SpCudaUtils::GetDefaultNbStreams()) {
    if(SpCudaUtils::GetNbDevices() < nbCudaWorkers){
        std::cout << "[SPECX] The number of devices asked ("
                  << nbCudaWorkers << ") is above the real number of devices ("
                  << SpCudaUtils::GetNbDevices() << ")" << std::endl;
        std::cout << "[SPECX] The real number will be used instead." << std::endl;
        nbCudaWorkers = SpCudaUtils::GetNbDevices();
    }

    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbCpuWorkers + nbWorkerPerCudas*nbCudaWorkers);

    for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::CPU_WORKER));
    }

    for(int idxCuda = 0; idxCuda < nbCudaWorkers; ++idxCuda) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerCudas; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::CUDA_WORKER));
            res.back()->cudaData.init(idxCuda);
        }
    }

    return res;
}

template <class ... Args>
static auto TeamOfCpuGpuWorkers(Args&& ... args) {
    return TeamOfCpuCudaWorkers(std::forward<Args>(args)...);
}
#endif
#ifdef SPECX_COMPILE_WITH_HIP
static small_vector<std::unique_ptr<SpWorker>> TeamOfHipWorkers(const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             int nbHipWorkers = SpHipUtils::GetNbDevices()) {
    if(SpHipUtils::GetNbDevices() < nbHipWorkers){
        std::cout << "[SPECX] The number of devices asked ("
                  << nbHipWorkers << ") is above the real number of devices ("
                  << SpHipUtils::GetNbDevices() << ")" << std::endl;
        std::cout << "[SPECX] The real number will be used instead." << std::endl;
        nbHipWorkers = SpHipUtils::GetNbDevices();
    }

    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbWorkerPerHips*nbHipWorkers);

    for(int idxHip = 0; idxHip < nbHipWorkers; ++idxHip) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerHips; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::HIP_WORKER));
            res.back()->hipData.init(idxHip);
        }
    }

    return res;
}

static small_vector<std::unique_ptr<SpWorker>> TeamOfCpuHipWorkers(const int nbCpuWorkers = SpUtils::DefaultNumThreads(),
                                             const int nbWorkerPerHips = SpHipUtils::GetDefaultNbStreams(),
                                             int nbHipWorkers = SpHipUtils::GetNbDevices()) {
    if(SpHipUtils::GetNbDevices() < nbHipWorkers){
        std::cout << "[SPECX] The number of devices asked ("
                  << nbHipWorkers << ") is above the real number of devices ("
                  << SpHipUtils::GetNbDevices() << ")" << std::endl;
        std::cout << "[SPECX] The real number will be used instead." << std::endl;
        nbHipWorkers = SpHipUtils::GetNbDevices();
    }

    small_vector<std::unique_ptr<SpWorker>> res;
    res.reserve(nbCpuWorkers + nbWorkerPerHips*nbHipWorkers);

    for(int idxWorker = 0; idxWorker < nbCpuWorkers; ++idxWorker) {
        res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::CPU_WORKER));
    }

    for(int idxHip = 0; idxHip < nbHipWorkers; ++idxHip) {
        for(int idxWorker = 0; idxWorker < nbWorkerPerHips; ++idxWorker) {
            res.emplace_back(std::make_unique<SpWorker>(SpWorkerTypes::Type::HIP_WORKER));
            res.back()->hipData.init(idxHip);
        }
    }

    return res;
}

template <class ... Args>
static auto TeamOfCpuGpuWorkers(Args&& ... args) {
    return TeamOfCpuHipWorkers(std::forward<Args>(args)...);
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
