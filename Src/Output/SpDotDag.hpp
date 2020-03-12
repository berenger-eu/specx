///////////////////////////////////////////////////////////////////////////
// Spetabaru - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPDOTDAT_HPP
#define SPDOTDAT_HPP

#include <fstream>
#include <set>

#include "Tasks/SpAbstractTask.hpp"
#include "Utils/SpTimePoint.hpp"
#include "Utils/small_vector.hpp"


namespace SpDotDag {

inline void GenerateDot(const std::string& outputFilename, const std::list<SpAbstractTask*>& tasksFinished, bool printAccesses) {
    // dot -Tpdf /tmp/graph.dot -o /tmp/fichier.pdf
    std::ofstream outputWriter(outputFilename);

    if(outputWriter.is_open() == false){
        throw std::invalid_argument("Cannot open filename : " + outputFilename);
    }

    outputWriter << "digraph G {\n";
    
    small_vector<SpAbstractTask*> deps;

    for(const auto& atask : tasksFinished){
        atask->getDependences(&deps);

        std::set<SpAbstractTask*> alreadyExist;

        for(const auto& taskDep : deps){
            if(alreadyExist.find(taskDep) == alreadyExist.end()){
                outputWriter << "\t" << atask->getId() << " -> " << taskDep->getId() << "\n";
                alreadyExist.insert(taskDep);
            }
        }

        outputWriter << "\t" << atask->getId() << " [label=\"" << atask->getTaskName() << (atask->isTaskEnabled()?"":" (Disabled)");
        
        if(printAccesses) {
            outputWriter << std::endl << atask->getTaskBodyString() << "\"];\n";
        } else {
            outputWriter << "\"];\n";
        }
        
        deps.clear();
    }

    outputWriter << "}\n";
}

}


#endif
