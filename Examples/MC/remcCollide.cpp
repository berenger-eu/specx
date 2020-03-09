///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Utils/SpModes.hpp"
#include "Utils/SpUtils.hpp"

#include "Tasks/SpTask.hpp"
#include "Runtimes/SpRuntime.hpp"

#include "Random/SpMTGenerator.hpp"
#include "Utils/small_vector.hpp"

#include "mcglobal.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();

    const int NbLoops = 3;
    const int NbInnerLoops = 2;
    static_assert(NbInnerLoops <= NbLoops, "Nb inner loops cannot be greater than Nb Loops");

    const int NbDomains = 5;
    const int NbParticlesPerDomain = 10;
    const double BoxWidth = 1;
    const double displacement = 0.00001;

    const int NbReplicas = 5;
    std::array<double,NbReplicas> betas;
    std::array<double,NbReplicas> temperatures;

    const double MinTemperature = 0.5;
    const double MaxTemperature = 1.5;

    for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
        betas[idxReplica] = 1;
        temperatures[idxReplica] = MinTemperature + double(idxReplica)*(MaxTemperature-MinTemperature)/double(NbReplicas-1);
    }

    std::array<size_t,NbReplicas> cptGeneratedSeq = {0};

    ///////////////////////////////////////////////////////////////////////////
    /// With a possible failure move
    ///////////////////////////////////////////////////////////////////////////

    const bool runSeqMove = false;
    const bool runTaskMove = false;
    const bool runSpecMove = false;

    std::array<double,NbReplicas> energySeq = {0};

    const int MaxIterationToMove = 5;
    const double collisionLimit = 0.00001;

    if(runSeqMove){
        std::array<SpMTGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;
        std::array<size_t,NbReplicas> replicaCptGenerated;

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            size_t& cptGenerated = replicaCptGenerated[idxReplica];

            randGen = SpMTGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
            always_assert(randGen.getNbValuesGenerated() == 3 * NbDomains * NbParticlesPerDomain);
            cptGenerated = randGen.getNbValuesGenerated();

            // Compute all
            energyAll = ComputeForAll(domains.data(), NbDomains);

            std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAll) << std::endl;
        }

        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                size_t& cptGenerated = replicaCptGenerated[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    int acceptedMove = 0;
                    int failedMove = 0;

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        // Move domain
                        Domain<double> movedDomain(0);
                        int nbAttempts = 0;
                        for(int idxMove = 0 ; idxMove < MaxIterationToMove ; ++idxMove){
                            movedDomain = MoveDomain<double>(domains[idxDomain], BoxWidth, displacement, randGen);
                            nbAttempts += 1;
                            if(DomainCollide(domains.data(), NbDomains, idxDomain, movedDomain, collisionLimit)){
                                movedDomain.clear();
                            }
                            else{
                                break;
                            }
                        }
                        always_assert(randGen.getNbValuesGenerated()-cptGenerated == size_t((3 * NbParticlesPerDomain)*nbAttempts));
                        randGen.skip((3 * NbParticlesPerDomain)*(MaxIterationToMove-nbAttempts));
                        cptGenerated = randGen.getNbValuesGenerated();

                        if(movedDomain.getNbParticles()){
                            always_assert(nbAttempts != MaxIterationToMove);
                            // Compute new energy
                            const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domains.data(), NbDomains,
                                                                                                    energyAll, idxDomain, movedDomain);

                            std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                            // Accept/reject
                            if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                // replace by new state
                                domains[idxDomain] = std::move(movedDomain);
                                energyAll.setColumn(idxDomain, deltaEnergy.second.data());
                                energyAll.setRow(idxDomain, deltaEnergy.second.data());
                                acceptedMove += 1;
                                std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                            }
                            else{
                                // leave as it is
                                std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                            }
                            always_assert(randGen.getNbValuesGenerated()-cptGenerated == 1);
                            cptGenerated = randGen.getNbValuesGenerated();
                        }
                        else{
                            randGen.skip(1);
                            failedMove += 1;
                        }
                    }
                    std::cout << "[" << idxReplica << "][" << idxLoop <<"] energy = " << GetEnergy(energyAll)
                              << " acceptance " << static_cast<double>(acceptedMove)/static_cast<double>(NbDomains) << std::endl;
                    std::cout << "[" << idxReplica << "][" << idxLoop <<"] failed moves = " << static_cast<double>(failedMove)/static_cast<double>(NbDomains) << std::endl;
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
                        std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                        nbExchanges += 1;
                    }
                }
                std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchanges)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
            }
        }

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            cptGeneratedSeq[idxReplica] = replicaRandGen[idxReplica].getNbValuesGenerated();

            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            energySeq[idxReplica] = GetEnergy(energyAll);
        }
    }
    if(runTaskMove){
        SpRuntime runtime(NumThreads);

        std::array<SpMTGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

            randGen = SpMTGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

            runtime.task(SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
            });
        }

        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    int* acceptedMove = new int(0);
                    int* failedMove = new int(0);

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        Domain<double>* movedDomain = new Domain<double>(0);
                        // Move domain
                        runtime.task(SpWrite(*movedDomain), SpRead(domains[idxDomain]),
                                     [&BoxWidth, &displacement, randGen](Domain<double>& movedDomainParam, const Domain<double>& domains_idxDomain) mutable {
                            movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                        });
                        randGen.skip(3*NbParticlesPerDomain);

                        runtime.task(SpWrite(energyAll),
                                     SpWrite(*movedDomain),
                                     SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                     SpWrite(domains[idxDomain]),
                                     SpAtomicWrite(*acceptedMove),
                                     SpAtomicWrite(*failedMove),
                                     [&Temperature, idxDomain, idxInnerLoop, idxReplica, &collisionLimit, &BoxWidth, &displacement, randGen](
                                     Matrix<double>& energyAllParam,
                                     Domain<double>& movedDomainParam,
                                     const SpArrayAccessor<const Domain<double>>& domainsParam,
                                     Domain<double>& domains_idxDomain,
                                     int& acceptedMoveParam,
                                     int& failedMoveParam) mutable {
                            if(DomainCollide(domainsParam, NbDomains, idxDomain, movedDomainParam, collisionLimit)){
                                int nbAttempt = 1;
                                for(int idxMove = 1 ; idxMove < MaxIterationToMove ; ++idxMove){
                                    movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                                    nbAttempt += 1;
                                    if(DomainCollide(domainsParam, NbDomains, idxDomain, movedDomainParam, collisionLimit)){
                                        movedDomainParam.clear();
                                    }
                                    else{
                                        break;
                                    }
                                }
                                randGen.skip((3*NbParticlesPerDomain)*(MaxIterationToMove-nbAttempt));
                            }

                            if(movedDomainParam.getNbParticles() != 0){
                                // Compute new energy
                                const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains,
                                                                                                        energyAllParam, idxDomain, movedDomainParam);

                                std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                                // Accept/reject
                                if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                    // replace by new state
                                    domains_idxDomain = std::move(movedDomainParam);
                                    energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                                    energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                                    acceptedMoveParam += 1;
                                    std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                                }
                                else{
                                    // leave as it is
                                    std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                                }
                            }
                            else{
                                failedMoveParam += 1;
                            }

                            delete &movedDomainParam;
                        });
                        randGen.skip(3*NbParticlesPerDomain*(MaxIterationToMove-1));
                        randGen.skip(1);
                    }

                    runtime.task(SpRead(energyAll), SpWrite(*acceptedMove), SpWrite(*failedMove),
                                 [idxInnerLoop, idxReplica](const Matrix<double>& energyAllParam, int& acceptedMoveParam, int& failedMoveParam){
                        std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                                  << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] failed moves = " << static_cast<double>(failedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        delete &acceptedMoveParam;
                        delete &failedMoveParam;
                    });
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                int* nbExchanges = new int(0);
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    SpMTGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                    auto& domains0 = replicaDomains[idxReplica];
                    Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                    auto& domains1 = replicaDomains[idxReplica+1];
                    Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                    runtime.task(SpWrite(domains0), SpWrite(energyAll0),
                                 SpWrite(domains1), SpWrite(energyAll1),
                                 SpWrite(*nbExchanges),
                                 [randGen0, idxReplica, &betas, idxLoop](small_vector<Domain<double>>& domains0Param,
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
                            std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                            nbExchangesParam += 1;
                        }
                    });
                    replicaRandGen[idxReplica].skip(1);
                }

                runtime.task(SpWrite(*nbExchanges),
                             [startExchangeIdx, idxLoop](int& nbExchangesParam){
                    std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                    delete &nbExchangesParam;
                });
            }
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            always_assert(runSeqMove == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            always_assert(runSeqMove == false || GetEnergy(energyAll) == energySeq[idxReplica]);
        }
    }
    if(runSpecMove){
        SpRuntime runtime(NumThreads);

        std::array<SpMTGenerator<double>,NbReplicas> replicaRandGen;
        std::array<small_vector<Domain<double>>,NbReplicas> replicaDomains;
        std::array<Matrix<double>,NbReplicas> replicaEnergyAll;

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
            auto& domains = replicaDomains[idxReplica];
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];

            randGen = SpMTGenerator<double>(0/*idxReplica*/);

            domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);

            runtime.task(SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [idxReplica](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START][" << idxReplica << "]" << " energy = " << GetEnergy(energyAllParam) << std::endl;
            });
        }

        for(int idxLoop = 0 ; idxLoop < NbLoops ; idxLoop += NbInnerLoops){
            const int NbInnerLoopsLimit = std::min(NbInnerLoops, NbLoops-idxLoop) + idxLoop;
            for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
                SpMTGenerator<double>& randGen = replicaRandGen[idxReplica];
                auto& domains = replicaDomains[idxReplica];
                Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
                const double& Temperature = temperatures[idxReplica];

                for(int idxInnerLoop = idxLoop ; idxInnerLoop < NbInnerLoopsLimit ; ++idxInnerLoop){
                    int* acceptedMove = new int(0);
                    int* failedMove = new int(0);

                    for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                        Domain<double>* movedDomain = new Domain<double>(0);
                        // Move domain
                        runtime.task(SpWrite(*movedDomain), SpRead(domains[idxDomain]),
                                     [&BoxWidth, &displacement, randGen](Domain<double>& movedDomainParam, const Domain<double>& domains_idxDomain) mutable {
                            movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                        });
                        randGen.skip(3*NbParticlesPerDomain);

                        runtime.task(SpMaybeWrite(energyAll),
                                     SpWrite(*movedDomain),
                                     SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                     SpMaybeWrite(domains[idxDomain]),
                                     SpAtomicWrite(*acceptedMove),
                                     SpAtomicWrite(*failedMove),
                                     [&Temperature, idxDomain, idxInnerLoop, idxReplica, &collisionLimit, &BoxWidth, &displacement, randGen](
                                     Matrix<double>& energyAllParam,
                                     Domain<double>& movedDomainParam,
                                     const SpArrayAccessor<const Domain<double>>& domainsParam,
                                     Domain<double>& domains_idxDomain,
                                     int& acceptedMoveParam,
                                     int& failedMoveParam) mutable {
                            if(DomainCollide(domainsParam, NbDomains, idxDomain, movedDomainParam, collisionLimit)){
                                int nbAttempt = 1;
                                for(int idxMove = 1 ; idxMove < MaxIterationToMove ; ++idxMove){
                                    movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                                    nbAttempt += 1;
                                    if(DomainCollide(domainsParam, NbDomains, idxDomain, movedDomainParam, collisionLimit)){
                                        movedDomainParam.clear();
                                    }
                                    else{
                                        break;
                                    }
                                }
                                randGen.skip((3*NbParticlesPerDomain)*(MaxIterationToMove-nbAttempt));
                            }

                            if(movedDomainParam.getNbParticles() != 0){
                                // Compute new energy
                                const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains,
                                                                                                        energyAllParam, idxDomain, movedDomainParam);

                                std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                                // Accept/reject
                                if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                                    // replace by new state
                                    domains_idxDomain = std::move(movedDomainParam);
                                    energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                                    energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                                    acceptedMoveParam += 1;
                                    std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t accepted " << std::endl;
                                    delete &movedDomainParam;
                                    return true;
                                }
                                else{
                                    // leave as it is
                                    std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] \t\t reject " << std::endl;
                                    delete &movedDomainParam;
                                    return false;
                                }
                            }
                            else{
                                failedMoveParam += 1;
                                delete &movedDomainParam;
                                return false;
                            }
                        });
                        randGen.skip(3*NbParticlesPerDomain*(MaxIterationToMove-1));
                        randGen.skip(1);
                    }

                    runtime.task(SpRead(energyAll), SpWrite(*acceptedMove), SpWrite(*failedMove),
                                 [idxInnerLoop, idxReplica](const Matrix<double>& energyAllParam, int& acceptedMoveParam, int& failedMoveParam){
                        std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] energy = " << GetEnergy(energyAllParam)
                                  << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        std::cout << "[" << idxReplica << "][" << idxInnerLoop <<"] failed moves = " << static_cast<double>(failedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                        delete &acceptedMoveParam;
                        delete &failedMoveParam;
                    });
                }
            }

            if(NbInnerLoopsLimit == NbInnerLoops){
                // Exchange
                int* nbExchanges = new int(0);
                const int startExchangeIdx = ((idxLoop/NbInnerLoops)&1);
                for(int idxReplica = startExchangeIdx ; idxReplica+1 < NbReplicas ; idxReplica += 2){
                    SpMTGenerator<double>& randGen0 = replicaRandGen[idxReplica];
                    auto& domains0 = replicaDomains[idxReplica];
                    Matrix<double>& energyAll0 = replicaEnergyAll[idxReplica];
                    auto& domains1 = replicaDomains[idxReplica+1];
                    Matrix<double>& energyAll1 = replicaEnergyAll[idxReplica+1];

                    runtime.task(SpMaybeWrite(domains0), SpMaybeWrite(energyAll0),
                                 SpMaybeWrite(domains1), SpMaybeWrite(energyAll1),
                                 SpAtomicWrite(*nbExchanges),
                                 [randGen0, idxReplica, &betas, idxLoop](small_vector<Domain<double>>& domains0Param,
                                 Matrix<double>& energyAll0Param, small_vector<Domain<double>>& domains1Param,
                                 Matrix<double>& energyAll1Param, int& nbExchangesParam) mutable -> bool {
                        const bool exchange = RemcAccept(GetEnergy(energyAll0Param),
                                                         GetEnergy(energyAll1Param),
                                                         betas[idxReplica],
                                                         betas[idxReplica+1],
                                                         randGen0);

                        if(exchange){
                            std::swap(domains0Param,domains1Param);
                            std::swap(energyAll0Param ,energyAll1Param);
                            std::cout << "[" << idxLoop <<"] exchange " << idxReplica << " <=> " << idxReplica+1 << std::endl;
                            nbExchangesParam += 1;
                            return true;
                        }
                        return false;
                    });
                    replicaRandGen[idxReplica].skip(1);
                }

                runtime.task(SpWrite(*nbExchanges),
                             [startExchangeIdx, idxLoop](int& nbExchangesParam){
                    std::cout << "[" << idxLoop <<"] exchange acceptance " << static_cast<double>(nbExchangesParam)/static_cast<double>(NbReplicas/2 - startExchangeIdx) << std::endl;
                    delete &nbExchangesParam;
                });
            }
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        for(int idxReplica = 0 ; idxReplica < NbReplicas ; ++idxReplica){
            always_assert(runSeqMove == false || cptGeneratedSeq[idxReplica] == replicaRandGen[idxReplica].getNbValuesGenerated());
            Matrix<double>& energyAll = replicaEnergyAll[idxReplica];
            always_assert(runSeqMove == false || GetEnergy(energyAll) == energySeq[idxReplica]);
        }
    }

    return 0;
}
