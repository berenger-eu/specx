#ifndef MCGLOBAL_HPP
#define MCGLOBAL_HPP

#include <memory>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>

#include "Utils/SpArrayAccessor.hpp"

template <class RealType>
RealType LennardJones(const RealType r_ij){
    const RealType d = 1;
    const RealType E0 = 1;

    const RealType d_r = d/r_ij;

    const RealType d_r_pw2 = d_r*d_r;
    const RealType d_r_pw4 = d_r_pw2*d_r_pw2;

    const RealType d_r_pw6 = d_r_pw2*d_r_pw4;
    const RealType d_r_pw12 = d_r_pw6*d_r_pw6;

    return 4 * E0 * (d_r_pw12 - d_r_pw6);
}


template <class RealType>
struct Particle{
    RealType x, y, z;

    RealType distance(const Particle& other) const{
        return sqrt((x-other.x)*(x-other.x) + (y-other.y)*(y-other.y) + (z-other.z)*(z-other.z));
    }
};


template <class RealType>
class Matrix {
    int nbRows;
    int nbCols;

    std::vector<RealType> data;

public:
    Matrix(const int inNbRows, const int inNbCols)
        : nbRows(inNbRows), nbCols(inNbCols){
        data.resize(inNbRows*inNbCols, RealType());
    }

    explicit Matrix()
        : nbRows(0), nbCols(0){
    }

    int getNbRows() const{
        return nbRows;
    }

    int getNbCols() const{
        return nbCols;
    }

    const RealType& value(const int idxRow, const int idxCol) const{
        return data[idxRow * nbCols + idxCol];
    }

    RealType& value(const int idxRow, const int idxCol) {
        return data[idxRow * nbCols + idxCol];
    }

    void setColumn(const int idxCol, const RealType line[]){
        for(int idxRow = 0 ; idxRow < nbRows ; ++idxRow){
            value(idxRow, idxCol) = line[idxRow];
        }
    }

    void setRow(const int idxRow, const RealType line[]){
        for(int idxCol = 0 ; idxCol < nbCols ; ++idxCol){
            value(idxRow, idxCol) = line[idxCol];
        }
    }
};


template <class RealType>
class Domain{
    std::vector<Particle<RealType>> particles;
public:
    explicit Domain(const int inNbParticles = 0){
        particles.resize(inNbParticles);
    }

    int getNbParticles() const{
        return static_cast<int>(particles.size());
    }

    const Particle<RealType>& getParticle(const int idxParticle) const{
        return particles[idxParticle];
    }

    Particle<RealType>& getParticle(const int idxParticle) {
        return particles[idxParticle];
    }

    void clear(){
        particles.clear();
    }
};

template <class RealType, class RandGenClass>
std::vector<Domain<RealType>> InitDomains(const int NbDomains, const int NbParticlesPerDomain, const RealType BoxWidth,
                                          RandGenClass& randGen){
    std::vector<Domain<RealType>> domains;
    domains.reserve(NbDomains);

    for(int idxDom = 0 ; idxDom < NbDomains ; ++idxDom){
        domains.emplace_back(NbParticlesPerDomain);

        for(int idxPart = 0 ; idxPart < domains[idxDom].getNbParticles() ; ++idxPart){
            Particle<RealType>& part = domains[idxDom].getParticle(idxPart);
            part.x += BoxWidth*randGen.getRand01();
            part.y += BoxWidth*randGen.getRand01();
            part.z += BoxWidth*randGen.getRand01();
        }
    }

    return domains;
}

template <class RealType, class RandGenClass>
Domain<RealType> MoveDomain(const Domain<RealType> domain, const RealType BoxWidth, const RealType displacement, RandGenClass& randGen){
    Domain<RealType> movedDomain = domain;

    for(int idxPart = 0 ; idxPart < movedDomain.getNbParticles() ; ++idxPart){
        Particle<RealType>& part = movedDomain.getParticle(idxPart);
        part.x += BoxWidth*displacement*(randGen.getRand01()-0.5);
        if(part.x < 0) part.x += BoxWidth;
        else if(BoxWidth <= part.x) part.x -= BoxWidth;

        part.y += BoxWidth*displacement*(randGen.getRand01()-0.5);
        if(part.y < 0) part.y += BoxWidth;
        else if(BoxWidth <= part.y) part.y -= BoxWidth;

        part.z += BoxWidth*displacement*(randGen.getRand01()-0.5);
        if(part.z < 0) part.z += BoxWidth;
        else if(BoxWidth <= part.z) part.z -= BoxWidth;
    }

    return movedDomain;
}

template <class RealType>
bool DomainCollide(const Domain<RealType> domains[], const int nbDomains, const int idxDomainToTest,
                   const Domain<RealType>& domainToTest, const RealType collisionLimit){
    RealType minDomain[3] = {std::numeric_limits<RealType>::max(),std::numeric_limits<RealType>::max(),std::numeric_limits<RealType>::max()};
    RealType maxDomain[3] = {std::numeric_limits<RealType>::min(),std::numeric_limits<RealType>::min(),std::numeric_limits<RealType>::min()};

    for(int idxPart = 0 ; idxPart < domainToTest.getNbParticles() ; ++idxPart){
        const Particle<RealType>& part = domainToTest.getParticle(idxPart);
        minDomain[0] = std::min(part.x, minDomain[0]);
        maxDomain[0] = std::max(part.x, minDomain[0]);

        minDomain[1] = std::min(part.y, minDomain[1]);
        maxDomain[1] = std::max(part.y, minDomain[1]);

        minDomain[2] = std::min(part.z, minDomain[2]);
        maxDomain[2] = std::max(part.z, minDomain[2]);
    }

    for(int idxDomain = 0 ; idxDomain < nbDomains ; ++idxDomain){
        if(idxDomain != idxDomainToTest){
            for(int idxPart = 0 ; idxPart < domains[idxDomain].getNbParticles() ; ++idxPart){
                const Particle<RealType>& part = domains[idxDomain].getParticle(idxPart);
                if(minDomain[0] <= part.x && part.x <= maxDomain[0]
                        && minDomain[1] <= part.y && part.y <= maxDomain[1]
                        && minDomain[2] <= part.z && part.z <= maxDomain[2]){

                    for(int idxPartTest = 0 ; idxPartTest < domainToTest.getNbParticles() ; ++idxPartTest){
                        const Particle<RealType>& partTest = domainToTest.getParticle(idxPart);
                        if(std::abs(partTest.x-part.x) <= collisionLimit
                                && std::abs(partTest.y-part.y) <= collisionLimit
                                && std::abs(partTest.z-part.z) <= collisionLimit){
                            return true;
                        }
                    }
                }
            }
        }
    }


    return false;
}


template <class RealType>
bool DomainCollide(const SpArrayAccessor<const Domain<RealType>>& domains, const int nbDomains, const int idxDomainToTest,
                   const Domain<RealType>& domainToTest, const RealType collisionLimit){
    assert(domains.getSize() == nbDomains);

    RealType minDomain[3] = {std::numeric_limits<RealType>::max(),std::numeric_limits<RealType>::max(),std::numeric_limits<RealType>::max()};
    RealType maxDomain[3] = {std::numeric_limits<RealType>::min(),std::numeric_limits<RealType>::min(),std::numeric_limits<RealType>::min()};

    for(int idxPart = 0 ; idxPart < domainToTest.getNbParticles() ; ++idxPart){
        const Particle<RealType>& part = domainToTest.getParticle(idxPart);
        minDomain[0] = std::min(part.x, minDomain[0]);
        maxDomain[0] = std::max(part.x, minDomain[0]);

        minDomain[1] = std::min(part.y, minDomain[1]);
        maxDomain[1] = std::max(part.y, minDomain[1]);

        minDomain[2] = std::min(part.z, minDomain[2]);
        maxDomain[2] = std::max(part.z, minDomain[2]);
    }

    for(int idxDomain = 0 ; idxDomain < nbDomains ; ++idxDomain){
        assert(domains.getIndexAt(idxDomain) != idxDomainToTest);

        for(int idxPart = 0 ; idxPart < domains.getAt(idxDomain).getNbParticles() ; ++idxPart){
            const Particle<RealType>& part = domains.getAt(idxDomain).getParticle(idxPart);
            if(minDomain[0] <= part.x && part.x <= maxDomain[0]
                    && minDomain[1] <= part.y && part.y <= maxDomain[1]
                    && minDomain[2] <= part.z && part.z <= maxDomain[2]){

                for(int idxPartTest = 0 ; idxPartTest < domainToTest.getNbParticles() ; ++idxPartTest){
                    const Particle<RealType>& partTest = domainToTest.getParticle(idxPart);
                    if(std::abs(partTest.x-part.x) <= collisionLimit
                            && std::abs(partTest.y-part.y) <= collisionLimit
                            && std::abs(partTest.z-part.z) <= collisionLimit){
                        return true;
                    }
                }
            }
        }
    }


    return false;
}

template <class RealType>
Matrix<RealType> ComputeForAll(const Domain<RealType> domains[], const int nbDomains){
    Matrix<RealType> allEnergy(nbDomains, nbDomains);

    for(int idxDomain1 = 0; idxDomain1 < nbDomains ; ++idxDomain1){
        for(int idxDomain2 = idxDomain1+1; idxDomain2 < nbDomains ; ++idxDomain2){
            RealType energy = 0;

            for(int idxPart1 = 0 ; idxPart1 < domains[idxDomain1].getNbParticles() ; ++idxPart1){
                for(int idxPart2 = 0 ; idxPart2 < domains[idxDomain2].getNbParticles() ; ++idxPart2){
                    const RealType r_ij = domains[idxDomain1].getParticle(idxPart1).distance(domains[idxDomain2].getParticle(idxPart2));
                    energy += LennardJones(r_ij);
                }
            }

            allEnergy.value(idxDomain1, idxDomain2) = energy;
            allEnergy.value(idxDomain2, idxDomain1) = energy;
        }
    }

    return allEnergy;
}

template <class RealType>
Matrix<RealType> ComputeForAll(const SpArrayAccessor<const Domain<RealType>>& domains, const int nbDomains){
    assert(domains.getSize() == nbDomains);

    Matrix<RealType> allEnergy(nbDomains, nbDomains);

    for(int idxDomain1 = 0; idxDomain1 < domains.getSize() ; ++idxDomain1){
        for(int idxDomain2 = idxDomain1+1; idxDomain2 < domains.getSize() ; ++idxDomain2){
            RealType energy = 0;

            for(int idxPart1 = 0 ; idxPart1 < domains.getAt(idxDomain1).getNbParticles() ; ++idxPart1){
                for(int idxPart2 = 0 ; idxPart2 < domains.getAt(idxDomain2).getNbParticles() ; ++idxPart2){
                    const RealType r_ij = domains.getAt(idxDomain1).getParticle(idxPart1).distance(domains.getAt(idxDomain2).getParticle(idxPart2));
                    energy += LennardJones(r_ij);
                }
            }

            allEnergy.value(int(domains.getIndexAt(idxDomain1)), int(domains.getIndexAt(idxDomain2))) = energy;
            allEnergy.value(int(domains.getIndexAt(idxDomain2)), int(domains.getIndexAt(idxDomain1))) = energy;
        }
    }

    return allEnergy;
}

template <class RealType>
std::pair<RealType,std::vector<RealType>> ComputeForOne(const Domain<RealType> domains[], const int nbDomains,
                       const Matrix<RealType>& allEnergy, const int idxTargetDomain,
                       const Domain<RealType>& movedDomain){

    RealType deltaEnergy = 0;
    std::vector<RealType> newEnergy(nbDomains, 0);

    for(int idxDomain2 = 0; idxDomain2 < nbDomains ; ++idxDomain2){
        if(idxDomain2 != idxTargetDomain){
            RealType energy = 0;

            for(int idxPart1 = 0 ; idxPart1 < movedDomain.getNbParticles() ; ++idxPart1){
                for(int idxPart2 = 0 ; idxPart2 < domains[idxDomain2].getNbParticles() ; ++idxPart2){
                    const RealType r_ij = movedDomain.getParticle(idxPart1).distance(domains[idxDomain2].getParticle(idxPart2));
                    energy += LennardJones(r_ij);
                }
            }

            newEnergy[idxDomain2] = energy;

            deltaEnergy = energy - allEnergy.value(idxTargetDomain, idxDomain2);
        }
    }

    return {deltaEnergy, std::move(newEnergy)};
}

template <class RealType>
std::pair<RealType,std::vector<RealType>> ComputeForOne(const SpArrayAccessor<const Domain<RealType>>& domains, const int nbDomains,
                       const Matrix<RealType>& allEnergy, const int idxTargetDomain,
                       const Domain<RealType>& movedDomain){
    RealType deltaEnergy = 0;
    std::vector<RealType> newEnergy(nbDomains, 0);

    for(int idxDomain2 = 0; idxDomain2 < domains.getSize() ; ++idxDomain2){
        assert(domains.getIndexAt(idxDomain2) != idxTargetDomain);
        RealType energy = 0;

        for(int idxPart1 = 0 ; idxPart1 < movedDomain.getNbParticles() ; ++idxPart1){
            for(int idxPart2 = 0 ; idxPart2 < domains.getAt(idxDomain2).getNbParticles() ; ++idxPart2){
                const RealType r_ij = movedDomain.getParticle(idxPart1).distance(domains.getAt(idxDomain2).getParticle(idxPart2));
                energy += LennardJones(r_ij);
            }
        }

        newEnergy[domains.getIndexAt(idxDomain2)] = energy;

        deltaEnergy = energy - allEnergy.value(idxTargetDomain, int(domains.getIndexAt(idxDomain2)));
    }

    return {deltaEnergy, std::move(newEnergy)};
}


template <class RealType, class RandGenClass>
bool MetropolisAccept(const RealType deltaEnergy, const RealType beta, RandGenClass& randGen){
    const RealType coef = deltaEnergy <= 0 ? 1 : std::exp(-beta * deltaEnergy);
    return randGen.getRand01() < coef;
}


template <class RealType>
RealType GetEnergy(const Matrix<RealType>& matrix){
    RealType energy = 0;
    for(int idxRow = 0; idxRow < matrix.getNbRows() ; ++idxRow){
        for(int idxCol = idxRow + 1 ; idxCol < matrix.getNbCols() ; ++idxCol){
            energy += matrix.value(idxRow, idxCol);
        }
    }
    return energy;
}


template <class RealType, class RandGenClass>
bool RemcAccept(const RealType energyS1, const RealType energyS2,
                const RealType betaS1, const RealType betaS2,
                                  RandGenClass& randGen) {
    const RealType deltaE = energyS2 - energyS1;
    const RealType B = betaS2 - betaS1;
    const RealType coef = std::min(1., std::exp(B * deltaE));
    return randGen.getRand01() < coef;
  }


#endif // MCGLOBAL_HPP
