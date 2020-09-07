///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"
#include "Utils/SpTimer.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

#include "Random/SpPhiloxGenerator.hpp"
#include "Utils/small_vector.hpp"

#include "mcglobal.hpp"

#include <sstream>  // for env var conversion
template <class VariableType>
inline const VariableType EnvStrToOther(const char* const str, const VariableType& defaultValue = VariableType()){
    if(str == nullptr || getenv(str) == nullptr){
        return defaultValue;
    }
    const char* strVal = getenv(str);
    std::istringstream iss(strVal,std::istringstream::in);
    VariableType value = defaultValue;
    iss >> value;
    if( /*iss.tellg()*/ iss.eof() ) return value;
    return defaultValue;
}

int main(){
    const int NumThreads = EnvStrToOther<int>("NBTHREADS", 25);//SpUtils::DefaultNumThreads();
    std::cout << "NumThreads = " << NumThreads << std::endl;

    const int NbInnerLoops = 5;
    const int NbLoops = std::max(NbInnerLoops,EnvStrToOther<int>("NBLOOPS", 5));
    std::cout << "NbLoops = " << NbLoops << std::endl;
    always_assert(NbInnerLoops <= NbLoops);

    const int NbDomains = 5;
    std::cout << "NbDomains = " << NbDomains << std::endl;
    const int NbParticlesPerDomain = EnvStrToOther<int>("NBPARTICLES", 2000);
    std::cout << "NbParticlesPerDomain = " << NbParticlesPerDomain << std::endl;
    const double BoxWidth = 1;
    std::cout << "BoxWidth = " << BoxWidth << std::endl;
    const double displacement = 0.00001;
    std::cout << "displacement = " << displacement << std::endl;

    const int NbReplicas = 5;
    std::array<double,NbReplicas> betas;
    std::array<double,NbReplicas> temperatures;

    const double MinTemperature = 0.5;
    const double MaxTemperature = 10.5;

    for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
        betas[idxReplica] = 1;
        temperatures[idxReplica] = MinTemperature + double(idxReplica)*(MaxTemperature-MinTemperature)/double(NbReplicas-1);
        std::cout << "Temperature = " << temperatures[idxReplica] << std::endl;
    }

    std::array<size_t,NbReplicas> cptGeneratedSeq = {0};

    const bool runSeq = (getenv("REMCNOSEQ") && strcmp(getenv("REMCNOSEQ"),"FALSE") == 0 ? false : true);
    const bool runTask = runSeq;
    const bool runSpec = true;
    const bool verbose = (getenv("VERBOSE") && strcmp(getenv("VERBOSE"),"TRUE") == 0 ? true : false);

    std::array<double,NbReplicas> energySeq = {0};

    SpTimer timerSeq;
    SpTimer timerTask;
    SpTimer timerSpec;
    SpTimer timerSpecAllReject;
    const int MaxidxConsecutiveSpec = 6;
    SpTimer timerSpecNoCons[MaxidxConsecutiveSpec];

    if(runSeq){
        std::array<SpPhiloxGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;
        std::array<size_t,NbReplicas> replicaCptGenerated;

        timerSeq.start();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            size_t& cptGenerated = replicaCptGenerated[idxReplica];

            randGen = SpPhiloxGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
            always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));
            cptGenerated = randGen.getNbValuesGenerated();

            // Compute all
            energyAll = ComputeForAll(domains.data(), NbDomains);

            std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
        }

        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                size_t& cptGenerated = replicaCptGenerated[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    int acceptedMove = 0;

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        // Move domain
                        Domain<double> movedDomain = MoveDomain<double>(domains[idxDomain], BoxWidth, displacement, randGen);
                        always_assert(randGen.getNbValuesGenerated()-cptGenerated == 3 * static_cast<size_t>(NbParticlesPerDomain));
                        cptGenerated = randGen.getNbValuesGenerated();

                        // Compute new energy
                        const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domains.data(), NbDomains,
                                                                                                energyAll, idxDomain, movedDomain);

                        if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                        // Accept/reject
                        if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                            // replace by new state
                            domains[idxDomain] = std::move(movedDomain);
                            energyAll.setColumn(idxDomain, deltaEnergy.second.data());
                            energyAll.setRow(idxDomain, deltaEnergy.second.data());
                            acceptedMove += 1;
                            if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                        }
                        else{
                            // leave as it is
                            if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                        }
                        always_assert(randGen.getNbValuesGenerated()-cptGenerated == 1);
                        cptGenerated = randGen.getNbValuesGenerated();
                    }
                    if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAll)
                                          << " acceptance " << static_cast<double>(acceptedMove)/static_cast<double>(NbDomains) << std::endl;
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                int nbExchanges = 0;
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    const bool exchange = RemcAccept(GetEnergy(replicaEnergyAll[idxReplica]),
                                                     GetEnergy(replicaEnergyAll[idxReplica+1]),
                            betas[idxReplica],
                            betas[idxReplica+1],
                            replicaRandGen[idxReplica]);
                    replicaCptGenerated[idxReplica] = replicaRandGen[idxReplica].getNbValuesGenerated();

                    if(exchange){
                        std::swap(replicaDomains[idxReplica],replicaDomains[idxReplica+1]);
                        std::swap(replicaEnergyAll[idxReplica],replicaEnergyAll[idxReplica+1]);
                        if(verbose) std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                        nbExchanges += 1;
                    }
                }
                if(verbose) std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchanges)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
            }
        }

        timerSeq.stop();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            cptGeneratedSeq[idxReplica] = replicaRandGen[idxReplica].getNbValuesGenerated();
        }

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            std::cout << "[END][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
            energySeq[idxReplica] = GetEnergy(energyAll);
        }
    }
    if(runTask){
        SpRuntime runtime(NumThreads);

        std::array<SpPhiloxGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

        timerTask.start();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

            randGen = SpPhiloxGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

            runtime.task(SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
            }).setTaskName("ComputeForAll");
        }


        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    int* acceptedMove = new int(0);

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        runtime.task(SpWrite(energyAll),
                                     SpWrite(domains[idxDomain]),
                                     SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                     SpAtomicWrite(*acceptedMove),
                                     [verbose, BoxWidth, displacement, Temperature, idxDomain, idxReplica, idxInnerLoop, randGen](
                                     Matrix<double>& energyAllParam,
                                     Domain<double>& domains_idxDomain,
                                     const SpArrayAccessor<const Domain<double>>& domainsParam,
                                     int& acceptedMoveParam) mutable {
                            Domain<double> movedDomainParam;
                            movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);

                            // Compute new energy
                            const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains,
                                                                                                    energyAllParam, idxDomain, movedDomainParam);

                            if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                            // Accept/reject
                            if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                // replace by new state
                                domains_idxDomain = std::move(movedDomainParam);
                                energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                                energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                                acceptedMoveParam += 1;
                                if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                            }
                            else{
                                // leave as it is
                                if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                            }
                        }).setTaskName("Accept ");
                        randGen.skip(3*NbParticlesPerDomain + 1);
                    }

                    runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                                 [verbose, idxReplica, idxInnerLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                        if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                                              << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        delete &acceptedMoveParam;
                    });
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                int* nbExchanges = new int(0);
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    SpPhiloxGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                    auto& domains0 = replicaDomains[idxReplica];
                    Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                    auto& domains1 = replicaDomains[idxReplica+1];
                    Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                    runtime.task(SpWrite(domains0), SpWrite(energyAll0),
                                 SpWrite(domains1), SpWrite(energyAll1),
                                 SpWrite(*nbExchanges),
                                 [verbose, randGen0, idxReplica, betas, idxLoop](small_vector<Domain<double>>& domains0Param,
                                 Matrix<double>& energyAll0Param, small_vector<Domain<double>>& domains1Param,
                                 Matrix<double>& energyAll1Param, int& nbExchangesParam) mutable{
                        const bool exchange = RemcAccept(GetEnergy(energyAll0Param),
                                                         GetEnergy(energyAll1Param),
                                                         betas[idxReplica],
                                                         betas[idxReplica+1],
                                randGen0);

                        if(exchange){
                            std::swap(domains0Param,domains1Param);
                            std::swap(energyAll0Param ,energyAll1Param);
                            if(verbose) std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                            nbExchangesParam += 1;
                        }
                    }).setTaskName("Exchange-replicas");
                    replicaRandGen[idxReplica].skip(1);
                }

                runtime.task(SpWrite(*nbExchanges),
                             [verbose, startExchangeIdx, idxLoop](int& nbExchangesParam){
                    if(verbose) std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                    delete &nbExchangesParam;
                });
            }
        }

        // Wait for task to finish
        runtime.waitAllTasks();
        timerTask.stop();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            always_assert(runSeq == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
        }

        runtime.generateDot("remc_nospec_without_collision.dot");
        runtime.generateTrace("remc_nospec_without_collision.svg");

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            std::cout << "[END][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;            
            always_assert(runSeq == false || GetEnergy(energyAll) == energySeq[idxReplica]);
        }
    }
    if(runSpec){
        SpRuntime runtime(NumThreads);

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        std::array<SpPhiloxGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

        timerSpec.start();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

            randGen = SpPhiloxGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

            runtime.task(SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
            }).setTaskName("ComputeForAll");
        }


        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    // TODO int* acceptedMove = new int(0);

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        runtime.task(
                                    SpMaybeWrite(energyAll),
                                    SpMaybeWrite(domains[idxDomain]),
                                    SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                    [verbose, BoxWidth, displacement, Temperature, idxDomain, idxReplica, idxInnerLoop, randGen](
                                    Matrix<double>& energyAllParam,
                                    Domain<double>& domains_idxDomain,
                                    const SpArrayAccessor<const Domain<double>>& domainsParam )mutable -> bool {
                            Domain<double> movedDomainParam;
                            movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);

                            // Compute new energy
                            const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains,
                                                                                                    energyAllParam, idxDomain, movedDomainParam);

                            if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                            // Accept/reject
                            if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                // replace by new state
                                domains_idxDomain = std::move(movedDomainParam);
                                energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                                energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                                // TODO acceptedMoveParam += 1;
                                if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                                return true;
                            }
                            else{
                                // leave as it is
                                if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                                return false;
                            }
                        }).setTaskName("Accept ");
                        randGen.skip(3*NbParticlesPerDomain + 1);
                    }

#define MODE1
#ifdef MODE1
                    runtime.task(SpWrite(energyAll),
                                 SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                                 [](Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
                    }).setTaskName("Sync");
#endif //MODE1

                    // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                    // TODO              [verbose, NbDomains, idxReplica, idxInnerLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                    // TODO     if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                    // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                    // TODO     delete &acceptedMoveParam;
                    // TODO });
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                // TODO int* nbExchanges = new int(0);
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    SpPhiloxGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                    auto& domains0 = replicaDomains[idxReplica];
                    Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                    auto& domains1 = replicaDomains[idxReplica+1];
                    Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                    runtime.task(
                                SpMaybeWriteArray(domains0.data(), SpArrayView(NbDomains)),
                                SpMaybeWrite(energyAll0),
                                SpMaybeWriteArray(domains1.data(), SpArrayView(NbDomains)),
                                SpMaybeWrite(energyAll1),
                                [verbose, randGen0, idxReplica, &betas, idxLoop](
                                SpArrayAccessor<Domain<double>>& domains0Param,
                                Matrix<double>& energyAll0Param,
                                SpArrayAccessor<Domain<double>>& domains1Param,
                                Matrix<double>& energyAll1Param) mutable -> bool {
                        const bool exchange = RemcAccept(GetEnergy(energyAll0Param),
                                                         GetEnergy(energyAll1Param),
                                                         betas[idxReplica],
                                                         betas[idxReplica+1],
                                randGen0);

                        if(exchange){
                            for(int idxDom = 0 ; idxDom < domains0Param.getSize() ; ++idxDom){
                                std::swap(domains0Param.getAt(idxDom),domains1Param.getAt(idxDom));
                            }
                            std::swap(energyAll0Param ,energyAll1Param);
                            if(verbose) std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                            // TODO nbExchangesParam += 1;
                            return true;
                        }
                        return false;
                    }).setTaskName("Exchange-replicas");
                    replicaRandGen[idxReplica].skip(1);
                }

                // TODO runtime.task(SpWrite(*nbExchanges),
                // TODO              [verbose, startExchangeIdx, NbReplicas, idxLoop](int& nbExchangesParam){
                // TODO     if(verbose) std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                // TODO     delete &nbExchangesParam;
                // TODO });
            }
        }

        // Wait for task to finish
        runtime.waitAllTasks();
        timerSpec.stop();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            always_assert(runSeq == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
        }

        runtime.generateDot("remc_spec_without_collision.dot");
        runtime.generateTrace("remc_spec_without_collision.svg");

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            std::cout << "[END][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
            always_assert(runSeq == false || GetEnergy(energyAll) == energySeq[idxReplica]);
        }
    }
    if(runSpec){
        for(int idxConsecutiveSpec = 0 ; idxConsecutiveSpec < MaxidxConsecutiveSpec ; ++idxConsecutiveSpec){
            SpRuntime runtime(NumThreads);

            runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
                return true;
            });

            std::array<SpPhiloxGenerator<double>,NbReplicas> replicaRandGen;
            std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
            std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

            timerSpecNoCons[idxConsecutiveSpec].start();

            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

                randGen = SpPhiloxGenerator<double>(0/*idxReplica*/);

                domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

                runtime.task(SpWrite(energyAll),
                             SpReadArray(domains.data(),SpArrayView(NbDomains)),
                             [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                    energyAllParam = ComputeForAll(domainsParam, NbDomains);
                    std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
                }).setTaskName("ComputeForAll");
            }


            int idxConsecutive[NbReplicas] = {0};

            for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
                const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
                for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                    SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                    auto& domains = replicaDomains[idxReplica];
                    Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                    const double& Temperature = temperatures[idxReplica];

                    for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                        // TODO int* acceptedMove = new int(0);

                        for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                            runtime.task(
                                        SpMaybeWrite(energyAll),
                                        SpMaybeWrite(domains[idxDomain]),
                                        SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                        [verbose, BoxWidth, displacement, Temperature, idxDomain, idxReplica, idxInnerLoop, randGen](
                                        Matrix<double>& energyAllParam,
                                        Domain<double>& domains_idxDomain,
                                        const SpArrayAccessor<const Domain<double>>& domainsParam)mutable -> bool {
                                Domain<double> movedDomainParam;
                                movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);

                                // Compute new energy
                                const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains,
                                                                                                        energyAllParam, idxDomain, movedDomainParam);

                                if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                                // Accept/reject
                                if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                    // replace by new state
                                    domains_idxDomain = std::move(movedDomainParam);
                                    energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                                    energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                                    // TODO acceptedMoveParam += 1;
                                    if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                                    return true;
                                }
                                else{
                                    // leave as it is
                                    if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                                    return false;
                                }
                            }).setTaskName("Accept ");
                            randGen.skip(3*NbParticlesPerDomain + 1);

#define MODE1
#ifdef MODE1
                            if(idxConsecutive[idxReplica]++ == idxConsecutiveSpec){
                                runtime.task(SpWrite(energyAll),
                                             SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                                             [](Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
                                }).setTaskName("Sync");
                                idxConsecutive[idxReplica] = 0;
                            }
#endif //MODE1
                        }

                        // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                        // TODO              [verbose, NbDomains, idxReplica, idxInnerLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                        // TODO     if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                        // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        // TODO     delete &acceptedMoveParam;
                        // TODO });
                    }
                }

                if(NbInnerLoopsLimit == NbInnerLoops){
                    // Exchange
                    // TODO int* nbExchanges = new int(0);
                    const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                    for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                        SpPhiloxGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                        auto& domains0 = replicaDomains[idxReplica];
                        Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                        auto& domains1 = replicaDomains[idxReplica+1];
                        Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                        runtime.task(
                                    SpMaybeWriteArray(domains0.data(), SpArrayView(NbDomains)),
                                    SpMaybeWrite(energyAll0),
                                    SpMaybeWriteArray(domains1.data(), SpArrayView(NbDomains)),
                                    SpMaybeWrite(energyAll1),
                                    [verbose, randGen0, idxReplica, &betas, idxLoop](
                                    SpArrayAccessor<Domain<double>>& domains0Param,
                                    Matrix<double>& energyAll0Param,
                                    SpArrayAccessor<Domain<double>>& domains1Param,
                                    Matrix<double>& energyAll1Param) mutable -> bool {
                            const bool exchange = RemcAccept(GetEnergy(energyAll0Param),
                                                             GetEnergy(energyAll1Param),
                                                             betas[idxReplica],
                                                             betas[idxReplica+1],
                                    randGen0);

                            if(exchange){
                                for(int idxDom = 0 ; idxDom < domains0Param.getSize() ; ++idxDom){
                                    std::swap(domains0Param.getAt(idxDom),domains1Param.getAt(idxDom));
                                }
                                std::swap(energyAll0Param ,energyAll1Param);
                                if(verbose) std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                                // TODO nbExchangesParam += 1;
                                return true;
                            }
                            return false;
                        }).setTaskName("Exchange-replicas");
                        replicaRandGen[idxReplica].skip(1);
                    }

                    // TODO runtime.task(SpWrite(*nbExchanges),
                    // TODO              [verbose, startExchangeIdx, NbReplicas, idxLoop](int& nbExchangesParam){
                    // TODO     if(verbose) std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                    // TODO     delete &nbExchangesParam;
                    // TODO });
                }
            }

            // Wait for task to finish
            runtime.waitAllTasks();
            timerSpecNoCons[idxConsecutiveSpec].stop();

            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                always_assert(runSeq == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
            }

            runtime.generateDot("remc_spec_without_collision_" + std::to_string(idxConsecutiveSpec) + ".dot");
            runtime.generateTrace("remc_spec_without_collision_" + std::to_string(idxConsecutiveSpec) + ".svg");


            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                std::cout << "[END][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
                always_assert(runSeq == false || GetEnergy(energyAll) == energySeq[idxReplica]);
            }
        }
    }
    if(runSpec){
        SpRuntime runtime(NumThreads);

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        std::array<SpPhiloxGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

        timerSpecAllReject.start();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

            randGen = SpPhiloxGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

            runtime.task(SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
            }).setTaskName("ComputeForAll");
        }


        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpPhiloxGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    // TODO int* acceptedMove = new int(0);

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        Domain<double>* tmpDomains[4] = {nullptr, nullptr, nullptr, nullptr};
                        for(int idxDomCp = 0 ; idxDomCp < 5 ; ++idxDomCp){
                            if(idxDomCp < idxDomain){
                                tmpDomains[idxDomCp] = &domains[idxDomCp];
                            }
                            else if(idxDomain < idxDomCp){
                                tmpDomains[idxDomCp-1] = &domains[idxDomCp];
                            }
                        }

                        runtime.task(
                                    SpMaybeWrite(energyAll),
                                    SpMaybeWrite(domains[idxDomain]),
                                    //SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                    SpRead(*tmpDomains[0]), SpRead(*tmpDomains[1]), SpRead(*tmpDomains[2]), SpRead(*tmpDomains[3]),
                                [verbose, BoxWidth, displacement, idxDomain, idxReplica, idxInnerLoop, randGen, &domains](
                                Matrix<double>& energyAllParam,
                                Domain<double>& domains_idxDomain,
                                const Domain<double>& /*d0*/, const Domain<double>& /*d1*/, const Domain<double>& /*d2*/, const Domain<double>& /*d3*/)mutable -> bool {
                            Domain<double> movedDomainParam;
                            movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);

                            // Compute new energy
                            const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domains.data(), NbDomains,
                                                                                                    energyAllParam, idxDomain, movedDomainParam);

                            if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                            // ALWAYS FAIL FOR TESTING // Accept/reject
                            // ALWAYS FAIL FOR TESTING if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                            // ALWAYS FAIL FOR TESTING     // replace by new state
                            // ALWAYS FAIL FOR TESTING     domains_idxDomain = std::move(movedDomainParam);
                            // ALWAYS FAIL FOR TESTING     energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                            // ALWAYS FAIL FOR TESTING     energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                            // ALWAYS FAIL FOR TESTING    // TODO acceptedMoveParam += 1;
                            // ALWAYS FAIL FOR TESTING     if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                            // ALWAYS FAIL FOR TESTING     return true;
                            // ALWAYS FAIL FOR TESTING }
                            // ALWAYS FAIL FOR TESTING else{
                            // ALWAYS FAIL FOR TESTING     // leave as it is
                            // ALWAYS FAIL FOR TESTING     if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                            return false;
                            // ALWAYS FAIL FOR TESTING }
                        }).setTaskName("Accept ");
                        randGen.skip(3*NbParticlesPerDomain + 1);
                    }

#ifdef MODE1
                    runtime.task(SpWrite(energyAll),
                                 SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                                 [](Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
                    }).setTaskName("Sync");
#endif //MODE1

                    // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                    // TODO              [verbose, NbDomains, idxReplica, idxInnerLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                    // TODO     if(verbose) std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                    // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                    // TODO     delete &acceptedMoveParam;
                    // TODO });
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                // TODO int* nbExchanges = new int(0);
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    SpPhiloxGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                    auto& domains0 = replicaDomains[idxReplica];
                    Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                    auto& domains1 = replicaDomains[idxReplica+1];
                    Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                    runtime.task(
                                SpMaybeWriteArray(domains0.data(), SpArrayView(NbDomains)),
                                SpMaybeWrite(energyAll0),
                                SpMaybeWriteArray(domains1.data(), SpArrayView(NbDomains)),
                                SpMaybeWrite(energyAll1),
                                [randGen0, idxReplica, &betas](
                                SpArrayAccessor<Domain<double>>& /*domains0Param*/,
                                Matrix<double>& energyAll0Param,
                                SpArrayAccessor<Domain<double>>& /*domains1Param*/,
                                Matrix<double>& energyAll1Param) mutable -> bool {
                        // ALWAYS FAIL FOR TESTING const bool exchange =
                        RemcAccept(GetEnergy(energyAll0Param),
                                   GetEnergy(energyAll1Param),
                                   betas[idxReplica],
                                   betas[idxReplica+1],
                                randGen0);

                        // ALWAYS FAIL FOR TESTING if(exchange){
                        // ALWAYS FAIL FOR TESTING     std::swap(domains0,domains1);
                        // ALWAYS FAIL FOR TESTING     std::swap(energyAll0Param ,energyAll1Param);
                        // ALWAYS FAIL FOR TESTING     if(verbose) std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                        // ALWAYS FAIL FOR TESTING     // TODO nbExchangesParam += 1;
                        // ALWAYS FAIL FOR TESTING     return true;
                        // ALWAYS FAIL FOR TESTING }
                        return false;
                    }).setTaskName("Exchange-replicas");
                    replicaRandGen[idxReplica].skip(1);
                }

                // TODO runtime.task(SpWrite(*nbExchanges),
                // TODO              [verbose, startExchangeIdx, NbReplicas, idxLoop](int& nbExchangesParam){
                // TODO     if(verbose) std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                // TODO     delete &nbExchangesParam;
                // TODO });
            }
        }

        // Wait for task to finish
        runtime.waitAllTasks();
        timerSpecAllReject.stop();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            always_assert(runSeq == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
        }

        runtime.generateDot("remc_spec_without_collision-all-reject.dot");
        runtime.generateTrace("remc_spec_without_collision-all-reject.svg");

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            std::cout << "[END][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
        }
    }

    std::cout << "Timings:" << std::endl;
    std::cout << "seq = " << timerSeq.getElapsed() << std::endl;
    std::cout << "task = " << timerTask.getElapsed() << std::endl;
    std::cout << "spec = " << timerSpec.getElapsed() << std::endl;
    std::cout << "spec-reject = " << timerSpecAllReject.getElapsed() << std::endl;
    for(int idxConsecutiveSpec = 0 ; idxConsecutiveSpec < MaxidxConsecutiveSpec ; ++idxConsecutiveSpec){
        std::cout << "spec-max-" << idxConsecutiveSpec << " = " << timerSpecNoCons[idxConsecutiveSpec].getElapsed() << std::endl;
    }

    return 0;
}
