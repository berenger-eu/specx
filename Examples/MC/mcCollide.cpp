///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "Data/SpDataAccessMode.hpp"
#include "Utils/SpUtils.hpp"

#include "Task/SpTask.hpp"
#include "Legacy/SpRuntime.hpp"

#include "Random/SpPhiloxGenerator.hpp"
#include "Utils/small_vector.hpp"

#include "mcglobal.hpp"

int main(){
    const int NumThreads = SpUtils::DefaultNumThreads();

    const int NbLoops = 1;
    const int NbDomains = 5;
    const int NbParticlesPerDomain = 2000;
    const double BoxWidth = 1;
    const double Temperature = 1;
    const double displacement = 0.00001;

    size_t cptGeneratedSeq = 0;  

    ///////////////////////////////////////////////////////////////////////////
    /// With a possible failure move
    ///////////////////////////////////////////////////////////////////////////

    const bool runSeqMove = false;
    const bool runTaskMove = false;
    const bool runSpecMove = false;

    double energySeq = 0;

    const int MaxIterationToMove = 5;
    const double collisionLimit = 0.00001;

    if(runSeqMove){
        SpPhiloxGenerator<double> randGen(0);

        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * NbDomains * NbParticlesPerDomain);
        size_t cptGenerated = randGen.getNbValuesGenerated();

        // Compute all
        Matrix<double> energyAll = ComputeForAll(domains.data(), NbDomains);

        std::cout << "[START] energy = " << GetEnergy(energyAll) << std::endl;

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
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

                    std::cout << "[" << idxLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                    // Accept/reject
                    if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                        // replace by new state
                        domains[idxDomain] = std::move(movedDomain);
                        energyAll.setColumn(idxDomain, deltaEnergy.second.data());
                        energyAll.setRow(idxDomain, deltaEnergy.second.data());
                        acceptedMove += 1;
                        std::cout << "[" << idxLoop <<"] \t\t accepted " << std::endl;
                    }
                    else{
                        // leave as it is
                        std::cout << "[" << idxLoop <<"] \t\t reject " << std::endl;
                    }
                    always_assert(randGen.getNbValuesGenerated()-cptGenerated == 1);
                    cptGenerated = randGen.getNbValuesGenerated();
                }
                else{
                    randGen.skip(1);
                    failedMove += 1;
                }
            }
            std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAll)
                      << " acceptance " << static_cast<double>(acceptedMove)/static_cast<double>(NbDomains) << std::endl;
            std::cout << "[" << idxLoop <<"] failed moves = " << static_cast<double>(failedMove)/static_cast<double>(NbDomains) << std::endl;
        }

        cptGeneratedSeq = randGen.getNbValuesGenerated();
        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            energySeq = GetEnergy(energyAllTmp);
        }
    }
    if(runTaskMove){
        SpRuntime runtime(NumThreads);

        SpPhiloxGenerator<double> randGen(0);

        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * NbDomains * NbParticlesPerDomain);

        // Compute all
        Matrix<double> energyAll(0,0);
        runtime.task(SpWrite(energyAll),
                     SpReadArray(domains.data(),SpArrayView(NbDomains)),
                     [](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
            energyAllParam = ComputeForAll(domainsParam, NbDomains);
            std::cout << "[START] energy = " << GetEnergy(energyAllParam) << std::endl;
        }).setTaskName("ComputeForAll");

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            int* acceptedMove = new int(0);
            int* failedMove = new int(0);

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                Domain<double>* movedDomain = new Domain<double>(0);
                // Move domain
                runtime.task(SpWrite(*movedDomain), SpRead(domains[idxDomain]),
                             [&BoxWidth, &displacement, randGen](Domain<double>& movedDomainParam, const Domain<double>& domains_idxDomain) mutable {
                    movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                }).setTaskName("MoveDomain -- "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(3*NbParticlesPerDomain);

                runtime.task(SpWrite(energyAll),
                             SpWrite(*movedDomain),
                             SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                             SpWrite(domains[idxDomain]),
                             SpParallelWrite(*acceptedMove),
                             SpParallelWrite(*failedMove),
                             [&Temperature, idxDomain, idxLoop, &collisionLimit, &BoxWidth, &displacement, randGen](
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

                        std::cout << "[" << idxLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                        // Accept/reject
                        if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                            // replace by new state
                            domains_idxDomain = std::move(movedDomainParam);
                            energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                            energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                            acceptedMoveParam += 1;
                            std::cout << "[" << idxLoop <<"] \t\t accepted " << std::endl;
                        }
                        else{
                            // leave as it is
                            std::cout << "[" << idxLoop <<"] \t\t reject " << std::endl;
                        }
                    }
                    else{
                        failedMoveParam += 1;
                    }

                    delete &movedDomainParam;
                }).setTaskName("Accept -- "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(3*NbParticlesPerDomain*(MaxIterationToMove-1));
                randGen.skip(1);
            }

            runtime.task(SpRead(energyAll), SpWrite(*acceptedMove), SpWrite(*failedMove),
                         [idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam, int& failedMoveParam){
                std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                std::cout << "[" << idxLoop <<"] failed moves = " << static_cast<double>(failedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                delete &acceptedMoveParam;
                delete &failedMoveParam;
            }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        always_assert(runSeqMove == false || cptGeneratedSeq == randGen.getNbValuesGenerated());
        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            always_assert(runSeqMove == false || GetEnergy(energyAllTmp) == energySeq);
        }

        runtime.generateDot("/tmp/no_spec_with_collision.dot");
        runtime.generateTrace("/tmp/no_spec_with_collision.svg");
    }
    if(runSpecMove){
        SpRuntime runtime(NumThreads);

        SpPhiloxGenerator<double> randGen(0);

        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * NbDomains * NbParticlesPerDomain);

        // Compute all
        Matrix<double> energyAll(0,0);
        runtime.task(SpPriority(0), SpWrite(energyAll),
                     SpReadArray(domains.data(),SpArrayView(NbDomains)),
                     [](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
            energyAllParam = ComputeForAll(domainsParam, NbDomains);
            std::cout << "[START] energy = " << GetEnergy(energyAllParam) << std::endl;
        }).setTaskName("ComputeForAll");

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            int* acceptedMove = new int(0);
            int* failedMove = new int(0);

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                Domain<double>* movedDomain = new Domain<double>(0);
                // Move domain
                runtime.task(SpWrite(*movedDomain), SpRead(domains[idxDomain]),
                             [&BoxWidth, &displacement, randGen](Domain<double>& movedDomainParam, const Domain<double>& domains_idxDomain) mutable {
                    movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                }).setTaskName("MoveDomain -- "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(3*NbParticlesPerDomain);

                runtime.task(SpPotentialWrite(energyAll),
                             SpWrite(*movedDomain),
                             SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                             SpPotentialWrite(domains[idxDomain]),
                             SpParallelWrite(*acceptedMove),
                             SpParallelWrite(*failedMove),
                             [&Temperature, idxDomain, idxLoop, &collisionLimit, &BoxWidth, &displacement, randGen](
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

                        std::cout << "[" << idxLoop <<"] \t delta energy = " << deltaEnergy.first << std::endl;

                        // Accept/reject
                        if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                            // replace by new state
                            domains_idxDomain = std::move(movedDomainParam);
                            energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                            energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                            acceptedMoveParam += 1;
                            std::cout << "[" << idxLoop <<"] \t\t accepted " << std::endl;
                            delete &movedDomainParam;
                            return true;
                        }
                        else{
                            // leave as it is
                            std::cout << "[" << idxLoop <<"] \t\t reject " << std::endl;
                            delete &movedDomainParam;
                            return false;
                        }
                    }
                    else{
                        failedMoveParam += 1;
                        delete &movedDomainParam;
                        return false;
                    }
                }).setTaskName("PotentialTask -- "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(3*NbParticlesPerDomain*(MaxIterationToMove-1));
                randGen.skip(1);
            }

            runtime.task(SpRead(energyAll), SpWrite(*acceptedMove), SpWrite(*failedMove),
                         [idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam, int& failedMoveParam){
                std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                std::cout << "[" << idxLoop <<"] failed moves = " << static_cast<double>(failedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                delete &acceptedMoveParam;
                delete &failedMoveParam;
            }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        always_assert(runSeqMove == false || cptGeneratedSeq == randGen.getNbValuesGenerated());
        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            always_assert(runSeqMove == false || GetEnergy(energyAllTmp) == energySeq);
        }

        runtime.generateDot("/tmp/spec_with_collision.dot");
        runtime.generateTrace("/tmp/spec_with_collision.svg");
    }


    return 0;
}
