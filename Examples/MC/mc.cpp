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

#define MODE1

int main(){
    const int NumThreads = EnvStrToOther<int>("NBTHREADS", 4);//SpUtils::DefaultNumThreads();
    std::cout << "NumThreads = " << NumThreads << std::endl;

    int NbLoops = EnvStrToOther<int>("NBLOOPS", 10);
    std::cout << "NbLoops = " << NbLoops << std::endl;
    const int NbDomains = 5;
    std::cout << "NbDomains = " << NbDomains << std::endl;
    const int NbParticlesPerDomain = EnvStrToOther<int>("NBPARTICLES", 2000);
    std::cout << "NbParticlesPerDomain = " << NbParticlesPerDomain << std::endl;
    const double BoxWidth = 1;
    std::cout << "BoxWidth = " << BoxWidth << std::endl;
    const double Temperature = 1;
    std::cout << "Temperature = " << Temperature << std::endl;
    const double displacement = 0.00001;
    std::cout << "displacement = " << displacement << std::endl;

    size_t cptGeneratedSeq = 0;

    const bool runSeq = true;
    const bool runTask = true;
    const bool runSpec = true;
    const bool verbose = (getenv("VERBOSE") && strcmp(getenv("VERBOSE"),"TRUE") == 0 ? true : false);

    double energySeq = 0;

    SpTimer timerSeq;
    SpTimer timerTask;
    SpTimer timerSpec;
    SpTimer timerSpecAllReject;
    const int MaxidxConsecutiveSpec = 6;
    SpTimer timerSpecNoCons[MaxidxConsecutiveSpec];

    if(runSeq){
        SpPhiloxGenerator<double> randGen(0);

        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));
        size_t cptGenerated = randGen.getNbValuesGenerated();

        timerSeq.start();
        // Compute all
        Matrix<double> energyAll = ComputeForAll(domains.data(), NbDomains);

        std::cout << "[START] energy = " << GetEnergy(energyAll) << std::endl;

        int totalAcceptedMove = 0;

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            int acceptedMove = 0;

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                // Move domain
                Domain<double> movedDomain = MoveDomain<double>(domains[idxDomain], BoxWidth, displacement, randGen);
                always_assert(randGen.getNbValuesGenerated()-cptGenerated == 3 * static_cast<size_t>(NbParticlesPerDomain));
                cptGenerated = randGen.getNbValuesGenerated();

                // Compute new energy
                const std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domains.data(), NbDomains,
                                                                                        energyAll, idxDomain, movedDomain);

                if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t delta energy = " << deltaEnergy.first << std::endl;

                // Accept/reject
                if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                    // replace by new state
                    domains[idxDomain] = std::move(movedDomain);
                    energyAll.setColumn(idxDomain, deltaEnergy.second.data());
                    energyAll.setRow(idxDomain, deltaEnergy.second.data());
                    acceptedMove += 1;
                    if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t accepted " << std::endl;
                }
                else{
                    // leave as it is
                    if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t reject " << std::endl;
                }
                always_assert(randGen.getNbValuesGenerated()-cptGenerated == 1);
                cptGenerated = randGen.getNbValuesGenerated();
            }

            totalAcceptedMove += acceptedMove;

            if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAll)
                      << " acceptance " << static_cast<double>(acceptedMove)/static_cast<double>(NbDomains) << std::endl;
        }

        cptGeneratedSeq = randGen.getNbValuesGenerated();
        timerSeq.stop();

        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            std::cout << "[End] energy = " << GetEnergy(energyAllTmp) << ", Accepted moves = "
                      << 100. * static_cast<double>(totalAcceptedMove)/static_cast<double>(NbDomains*NbLoops) << "%" << std::endl;
            energySeq = GetEnergy(energyAllTmp);
        }
    }
    if(runTask){
        SpRuntime runtime(NumThreads);

        SpPhiloxGenerator<double> randGen(0);

        timerTask.start();
        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));

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

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                runtime.task(SpWrite(energyAll),
                             SpWrite(domains[idxDomain]),
                             SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                             SpAtomicWrite(*acceptedMove),
                             [verbose, BoxWidth, displacement, Temperature, idxDomain, idxLoop, randGen](
                             Matrix<double>& energyAllParam,
                             Domain<double>& domains_idxDomain,
                             const SpArrayAccessor<const Domain<double>>& domainsParam,
                             int& acceptedMoveParam) mutable {
                    // Move domain
                    Domain<double> movedDomainParam;
                    movedDomainParam = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);

                    std::pair<double,small_vector<double>> deltaEnergyParam = ComputeForOne(domainsParam, NbDomains, energyAllParam, idxDomain, movedDomainParam);
                    if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t delta energy = " << deltaEnergyParam.first << std::endl;

                    // Accept/reject
                    if(MetropolisAccept(deltaEnergyParam.first, Temperature, randGen)){
                        // replace by new state
                        domains_idxDomain = std::move(movedDomainParam);
                        energyAllParam.setColumn(idxDomain, deltaEnergyParam.second.data());
                        energyAllParam.setRow(idxDomain, deltaEnergyParam.second.data());
                        acceptedMoveParam += 1;
                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t accepted " << std::endl;
                    }
                    else{
                        // leave as it is
                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t reject " << std::endl;
                    }
                }).setTaskName("Accept -- "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(3*NbParticlesPerDomain + 1);
            }

            runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                         [verbose, idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                delete &acceptedMoveParam;
            }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
        }

        // Wait for task to finish
        runtime.waitAllTasks();
        timerTask.stop();

        always_assert(runSeq == false || cptGeneratedSeq == randGen.getNbValuesGenerated());

        runtime.generateDot("mc_nospec_without_collision.dot");
        runtime.generateTrace("mc_nospec_without_collision.svg");

        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            std::cout << "[End] energy = " << GetEnergy(energyAllTmp) << std::endl;
            always_assert(runSeq == false || GetEnergy(energyAllTmp) == energySeq);
        }
    }
    if(runSpec){
        SpRuntime runtime(NumThreads);

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        SpPhiloxGenerator<double> randGen(0);

        timerSpec.start();
        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));

        // Compute all
        Matrix<double> energyAll(0,0);
        runtime.task(SpPriority(0), SpWrite(energyAll),
                     SpReadArray(domains.data(),SpArrayView(NbDomains)),
                     [](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
            energyAllParam = ComputeForAll(domainsParam, NbDomains);
            std::cout << "[START] energy = " << GetEnergy(energyAllParam) << std::endl;
        }).setTaskName("ComputeForAll");

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            // TODO int* acceptedMove = new int(0);

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                runtime.task(
                             SpMaybeWrite(energyAll),
                             SpMaybeWrite(domains[idxDomain]),
                             SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                             [verbose, idxDomain, idxLoop, BoxWidth, displacement, Temperature, randGen](
                             Matrix<double>& energyAllParam,
                             Domain<double>& domains_idxDomain,
                             const SpArrayAccessor<const Domain<double>>& domainsParam
                             ) mutable {
                    Domain<double> movedDomain;
                    movedDomain = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                    // Compute new energy
                    std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains, energyAllParam, idxDomain, movedDomain);

                    if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t delta energy = " << deltaEnergy.first << std::endl;

                    // Accept/reject
                    bool accepted;
                    if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                        // replace by new state
                        domains_idxDomain = std::move(movedDomain);
                        energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                        energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                        // TODO acceptedMoveParam += 1;
                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t accepted " << std::endl;
                        accepted = true;
                    }
                    else{
                        // leave as it is
                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t reject " << std::endl;
                        accepted = false;
                    }
                    // TODO if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                    // TODO          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                    return accepted;
                }).setTaskName("Accept -- Potential "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(1 + 3*NbParticlesPerDomain);
            }

#ifdef MODE1
            runtime.task(SpWrite(energyAll), SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                [](const Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
            });
#endif //MODE1

            // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
            // TODO              [verbose, NbDomains, idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
            // TODO     if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
            // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
            // TODO     delete &acceptedMoveParam;
            // TODO }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        timerSpec.stop();

        always_assert(runSeq == false || cptGeneratedSeq == randGen.getNbValuesGenerated());

        runtime.generateDot("mc_spec_without_collision.dot");
        runtime.generateTrace("mc_spec_without_collision.svg");

        {
            Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
            std::cout << "[End] energy = " << GetEnergy(energyAllTmp) << std::endl;
            always_assert(runSeq == false || GetEnergy(energyAllTmp) == energySeq);
        }
    }
    if(runSpec){
        for(int idxConsecutiveSpec = 0 ; idxConsecutiveSpec < MaxidxConsecutiveSpec ; ++idxConsecutiveSpec){
            SpRuntime runtime(NumThreads);

            runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
                return true;
            });

            SpPhiloxGenerator<double> randGen(0);

            timerSpecNoCons[idxConsecutiveSpec].start();
            small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
            always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));

            // Compute all
            Matrix<double> energyAll(0,0);
            runtime.task(SpPriority(0), SpWrite(energyAll),
                         SpReadArray(domains.data(),SpArrayView(NbDomains)),
                         [](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
                energyAllParam = ComputeForAll(domainsParam, NbDomains);
                std::cout << "[START] energy = " << GetEnergy(energyAllParam) << std::endl;
            }).setTaskName("ComputeForAll");

            int idxConsecutive = 0;

            for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                // TODO int* acceptedMove = new int(0);

                for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                    runtime.task(
                                 SpMaybeWrite(energyAll),
                                 SpMaybeWrite(domains[idxDomain]),
                                 SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                                 [verbose, idxDomain, idxLoop, BoxWidth, displacement, Temperature, randGen](
                                 Matrix<double>& energyAllParam,
                                 Domain<double>& domains_idxDomain,
                                 const SpArrayAccessor<const Domain<double>>& domainsParam) mutable {
                        Domain<double> movedDomain;
                        movedDomain = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                        // Compute new energy
                        std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains, energyAllParam, idxDomain, movedDomain);

                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t delta energy = " << deltaEnergy.first << std::endl;

                        // Accept/reject
                        bool accepted;
                        if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                            // replace by new state
                            domains_idxDomain = std::move(movedDomain);
                            energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                            energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                            // TODO acceptedMoveParam += 1;
                            if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t accepted " << std::endl;
                            accepted = true;
                        }
                        else{
                            // leave as it is
                            if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t reject " << std::endl;
                            accepted = false;
                        }
                        // TODO if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                        // TODO          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;

                        return accepted;
                    }).setTaskName("Accept -- Potential "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                    randGen.skip(1 + 3*NbParticlesPerDomain);
    #ifdef MODE1
                    if(idxConsecutive++ == idxConsecutiveSpec){
                        runtime.task(SpWrite(energyAll), SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                                     [](Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
                        });
                        idxConsecutive = 0;
                    }
    #endif
                }

                // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
                // TODO              [verbose, NbDomains, idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
                // TODO     if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
                // TODO     delete &acceptedMoveParam;
                // TODO }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
            }

            // Wait for task to finish
            runtime.waitAllTasks();

            timerSpecNoCons[idxConsecutiveSpec].stop();

            always_assert(runSeq == false || cptGeneratedSeq == randGen.getNbValuesGenerated());

            runtime.generateDot("mc_spec_without_collision_" + std::to_string(idxConsecutiveSpec) + ".dot");
            runtime.generateTrace("mc_spec_without_collision_" + std::to_string(idxConsecutiveSpec) + ".svg");

            {
                Matrix<double> energyAllTmp = ComputeForAll(domains.data(), NbDomains);
                std::cout << "[End] energy = " << GetEnergy(energyAllTmp) << std::endl;
                always_assert(runSeq == false || GetEnergy(energyAllTmp) == energySeq);
            }
        }
    }
    if(runSpec){
        SpRuntime runtime(NumThreads);

        runtime.setSpeculationTest([](const int /*inNbReadyTasks*/, const SpProbability& /*inProbability*/) -> bool{
            return true;
        });

        SpPhiloxGenerator<double> randGen(0);

        timerSpecAllReject.start();

        small_vector<Domain<double>> domains = InitDomains<double>(NbDomains, NbParticlesPerDomain, BoxWidth, randGen);
        always_assert(randGen.getNbValuesGenerated() == 3 * static_cast<size_t>(NbDomains) * static_cast<size_t>(NbParticlesPerDomain));

        // Compute all
        Matrix<double> energyAll(0,0);
        runtime.task(SpPriority(0), SpWrite(energyAll),
                     SpReadArray(domains.data(),SpArrayView(NbDomains)),
                     [](Matrix<double>& energyAllParam, const SpArrayAccessor<const Domain<double>>& domainsParam){
            energyAllParam = ComputeForAll(domainsParam, NbDomains);
            std::cout << "[START] energy = " << GetEnergy(energyAllParam) << std::endl;
        }).setTaskName("ComputeForAll");

        for(int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            // TODO int* acceptedMove = new int(0);

            for(int idxDomain = 0 ; idxDomain < NbDomains ; ++idxDomain){
                runtime.task(
                             SpMaybeWrite(energyAll),
                             SpMaybeWrite(domains[idxDomain]),
                             SpReadArray(domains.data(),SpArrayView(NbDomains).removeItem(idxDomain)),
                             [verbose, idxDomain, idxLoop, BoxWidth, displacement, randGen](
                             Matrix<double>& energyAllParam,
                             Domain<double>& domains_idxDomain,
                             const SpArrayAccessor<const Domain<double>>& domainsParam) mutable {
                    Domain<double> movedDomain;
                    movedDomain = MoveDomain<double>(domains_idxDomain, BoxWidth, displacement, randGen);
                    // Compute new energy
                    std::pair<double,small_vector<double>> deltaEnergy = ComputeForOne(domainsParam, NbDomains, energyAllParam, idxDomain, movedDomain);

                    if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t delta energy = " << deltaEnergy.first << std::endl;

                    // Accept/reject
                    bool accepted;
                    // ALWAYS FAIL FOR TESTING if(MetropolisAccept(deltaEnergy.first, Temperature, randGen)){
                    // ALWAYS FAIL FOR TESTING     // replace by new state
                    // ALWAYS FAIL FOR TESTING     domains_idxDomain = std::move(movedDomain);
                    // ALWAYS FAIL FOR TESTING     energyAllParam.setColumn(idxDomain, deltaEnergy.second.data());
                    // ALWAYS FAIL FOR TESTING     energyAllParam.setRow(idxDomain, deltaEnergy.second.data());
                    // ALWAYS FAIL FOR TESTING     // TODO acceptedMoveParam += 1;
                    // ALWAYS FAIL FOR TESTING     if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t accepted " << std::endl;
                    // ALWAYS FAIL FOR TESTING     accepted = true;
                    // ALWAYS FAIL FOR TESTING }
                    // ALWAYS FAIL FOR TESTING else{
                        // leave as it is
                        if(verbose) std::cout << "[" << idxLoop <<"][" << idxDomain <<"]\t\t reject " << std::endl;
                        accepted = false;
                    // ALWAYS FAIL FOR TESTING }
                    // TODO if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
                    // TODO          << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;

                    return accepted;
                }).setTaskName("Accept -- Potential "+std::to_string(idxLoop)+"/"+std::to_string(idxDomain));
                randGen.skip(1 + 3*NbParticlesPerDomain);
            }

#ifdef MODE1
            runtime.task(SpWrite(energyAll), SpWriteArray(domains.data(),SpArrayView(NbDomains)),
                [](const Matrix<double>& /*energyAllParam*/, const SpArrayAccessor<Domain<double>>& /*domainsParam*/){
            });
#endif //MODE1

            // TODO runtime.task(SpRead(energyAll), SpWrite(*acceptedMove),
            // TODO              [verbose, NbDomains, idxLoop](const Matrix<double>& energyAllParam, int& acceptedMoveParam){
            // TODO     if(verbose) std::cout << "[" << idxLoop <<"] energy = " << GetEnergy(energyAllParam)
            // TODO               << " acceptance " << static_cast<double>(acceptedMoveParam)/static_cast<double>(NbDomains) << std::endl;
            // TODO     delete &acceptedMoveParam;
            // TODO }).setTaskName("Energy-print -- "+std::to_string(idxLoop));
        }

        // Wait for task to finish
        runtime.waitAllTasks();

        timerSpecAllReject.stop();

        //always_assert(runSeq == false || cptGeneratedSeq == randGen.getNbValuesGenerated());
        //always_assert(runSeq == false || GetEnergy(energyAllTmp) == energySeq);

        runtime.generateDot("mc_spec_without_collision-all-reject.dot");
        runtime.generateTrace("mc_spec_without_collision-all-reject.svg");
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
