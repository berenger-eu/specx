///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPSVGTRACE_HPP
#define SPSVGTRACE_HPP

#include <fstream>
#include <cmath>
#include <iterator>
#include <unordered_map>

#include "Task/SpAbstractTask.hpp"
#include "Utils/SpTimePoint.hpp"
#include "Utils/small_vector.hpp"
#include "Random/SpMTGenerator.hpp"


namespace SpSvgTrace {

inline void GenerateTrace(const std::string& outputFilename, const std::list<SpAbstractTask*>& tasksFinished,
                          const SpTimePoint& startingTime, const bool showDependences) {
                              
    std::ofstream svgfile(outputFilename);

    if(svgfile.is_open() == false){
        throw std::invalid_argument("Cannot open filename : " + outputFilename);
    }
    
    const auto threadIds =
    [&]() {
        std::vector<long int> res;
        res.reserve(tasksFinished.size());
        std::transform(std::begin(tasksFinished), std::end(tasksFinished), std::back_inserter(res),
                      [](SpAbstractTask* task){
                        return task->getThreadIdComputer();
                      });
                      
        std::sort(std::begin(res), std::end(res));
        res.erase(std::unique(std::begin(res), std::end(res)), std::end(res));
        return res;
    }();
    
    const int nbThreads = static_cast<int>(threadIds.size());
    
    const auto threadIdsToVerticalSlotPosMap =
    [&]() {
        std::unordered_map<long int, long int> mapping;
        for(long int i=0; i < static_cast<long int>(threadIds.size()); i++) {
            mapping[threadIds[i]] = i+1;
        }
        return mapping;
    }();

    const long int vsizeperthread = std::max(100, std::min(200, 2000/nbThreads));
    const long int vmargin = 100;
    const long int threadstrock = 5;
    const long int vmarginthreadstats = 10;
    const long int vdim = nbThreads * (vsizeperthread+threadstrock) + 2*vmargin + vmarginthreadstats + 2*vsizeperthread;

    double duration = 0;
    for(const auto& atask : tasksFinished){
        duration = std::max(duration, startingTime.differenceWith(atask->getEndingTime()));
    }

    const long int hdimtime = std::max(2000, int(log(duration)*50));
    const long int hmargin = 50;
    const long int hdim = hdimtime + 2*hmargin;

    svgfile << "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n";
    svgfile << "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"" << hdim << "\" height=\"" << vdim << "\">\n";
    svgfile << "  <title>Execution trace</title>\n";
    svgfile << "  <desc>\n";
    svgfile << "    Specx traces for " << tasksFinished.size() << " tasks\n";
    svgfile << "  </desc>\n";
    svgfile << "\n";
    // Back
    svgfile << "  <rect width=\"" << hdim << "\" height=\"" << vdim << "\" x=\"0\" y=\"0\" fill=\"white\" />\n";


    svgfile << "  <line x1=\"" << hmargin << "\" y1=\"" << vmargin-10
            << "\" x2=\"" << hdimtime+hmargin << "\" y2=\"" << vmargin-10 << "\" stroke=\"" << "black" <<"\" stroke-width=\"" << "1" <<"\"/>\n";

    svgfile << "  <circle cx=\"" << hmargin << "\" cy=\"" << vmargin-10 << "\" r=\"" << "6" << "\" fill=\"" << "black" <<"\" />\n";
    svgfile << "  <circle cx=\"" << hdimtime+hmargin << "\" cy=\"" << vmargin-10 << "\" r=\"" << "6" << "\" fill=\"" << "black" << "\" />\n";

    for(double second = 1 ; second < duration ; second += 1.){
        svgfile << "  <line x1=\"" << static_cast<long  int>(double(hdimtime)*second/duration)+hmargin  << "\" y1=\"" << vmargin-14
                << "\" x2=\"" << static_cast<long  int>(double(hdimtime)*second/duration)+hmargin << "\" y2=\"" << vmargin-6 << "\" stroke=\"" << "black" <<"\" stroke-width=\"" << "1" <<"\"/>\n";
    }
    for(double second10 = 10 ; second10 < duration ; second10 += 10.){
        svgfile << "  <line x1=\"" << static_cast<long  int>(double(hdimtime)*second10/duration)+hmargin  << "\" y1=\"" << vmargin-18
                << "\" x2=\"" << static_cast<long  int>(double(hdimtime)*second10/duration)+hmargin << "\" y2=\"" << vmargin-2 << "\" stroke=\"" << "black" <<"\" stroke-width=\"" << "1" <<"\"/>\n";
    }

    {
        const std::string label = "Total time = " + std::to_string(duration) + " s";
        svgfile << "<text x=\"" << hmargin+hdimtime/2-int(label.size())*5 << "\" y=\"" << vmargin-20 <<
               "\" font-size=\"30\" fill=\"black\">" << label << "</text>\n";
    }

    for(auto idxThread : threadIds) {
        auto yPos = threadIdsToVerticalSlotPosMap.at(idxThread);
        svgfile << "  <rect width=\"" << hdimtime << "\" height=\"" << vsizeperthread
                << "\" x=\"" << hmargin << "\" y=\"" << (yPos-1)*(vsizeperthread+threadstrock) + vmargin << "\" style=\"fill:white;stroke:black;stroke-width:" << threadstrock << "\" />\n";

        const std::string label = "Thread " + std::to_string(idxThread);

        svgfile << "<text x=\"" << hmargin/2 << "\" y=\"" << (yPos-1)*(vsizeperthread+threadstrock) + vmargin + label.size()*30 - vsizeperthread/2 <<
                   "\" font-size=\"30\" fill=\"black\" transform=\"rotate(-90, " << hmargin/2 << " "
                << (yPos-1)*(vsizeperthread+threadstrock) + vmargin + label.size()*30 - vsizeperthread/2 << ")\">" << label << "</text>\n";
    }

    std::unordered_map<std::string, std::string> colors;

    for(const auto& atask : tasksFinished){
        const long int idxThreadComputer = atask->getThreadIdComputer();
        const long int ypos_start = (threadIdsToVerticalSlotPosMap.at(idxThreadComputer)-1)*(vsizeperthread+threadstrock) + threadstrock/2 + vmargin;
        const long int ypos_end = ypos_start + vsizeperthread - threadstrock;
        const double taskStartTime = startingTime.differenceWith(atask->getStartingTime());
        const double taskEndTime = startingTime.differenceWith(atask->getEndingTime());
        const long int xpos_start = static_cast<long  int>(double(hdimtime)*taskStartTime/duration) + hmargin;
        const long int xpos_end = std::max(xpos_start+1,static_cast<long  int>(double(hdimtime)*taskEndTime/duration) + hmargin);
        const long int taskstrocke = 2;

        std::string strForColor = atask->getTaskName();

        const std::size_t ddpos = strForColor.find("--");
        if(ddpos != std::string::npos){
            strForColor = strForColor.substr(0, ddpos);
        }
        if(atask->getTaskName().length() && atask->getTaskName().at(atask->getTaskName().length()-1) == '\''){
            strForColor.append("'");
        }

        if(colors.find(strForColor) == colors.end()){
            size_t hashname =  std::hash<std::string>()(strForColor);

            SpMTGenerator<> randEngine(hashname);
            const int colorR = int(randEngine.getRand01()*200) + 50;
            const int colorG = int(randEngine.getRand01()*200) + 50;
            const int colorB = int(randEngine.getRand01()*200) + 50;
            colors[strForColor] = std::string("rgb(")
                    + std::to_string(colorR) + ","
                    + std::to_string(colorG) + ","
                    + std::to_string(colorB) + ")";
        }

        svgfile << "<g>\n";
        svgfile << "    <title id=\"" << long(atask) << "\">" << SpUtils::ReplaceAllInString(SpUtils::ReplaceAllInString(SpUtils::ReplaceAllInString(atask->getTaskName(),"&", " REF "),"<","["),">","]")
                << " -- Duration " << atask->getStartingTime().differenceWith(atask->getEndingTime()) << "s"
                << " -- Enable = " << (atask->isTaskEnabled()?"TRUE":"FALSE") << "</title>\n";

        svgfile << "    <rect width=\"" << xpos_end-xpos_start << "\" height=\"" << ypos_end-ypos_start
                << "\" x=\"" << xpos_start << "\" y=\"" << ypos_start
                << "\" style=\"fill:" << colors[strForColor] << ";stroke:black" << ";stroke-width:" << taskstrocke << "\" />\n";

        svgfile << "</g>\n";
    }

    if(showDependences){
        svgfile << "<defs>\n"
                   "<marker id=\"arrow\" markerWidth=\"20\" markerHeight=\"20\" refX=\"20\" refY=\"6\" orient=\"auto\" markerUnits=\"strokeWidth\">\n"
                   "<path d=\"M0,0 L0,13 L20,6 z\" fill=\"gray\" />\n"
                   "</marker>\n"
                   "</defs>\n";

        small_vector<SpAbstractTask*> deps;

        for(const auto& atask : tasksFinished){
            atask->getDependences(&deps);
            const long int ypos_start = (threadIdsToVerticalSlotPosMap.at(atask->getThreadIdComputer())-1)*(vsizeperthread+threadstrock) + threadstrock/2 + vmargin + vsizeperthread/2;
            const double taskEndTime = startingTime.differenceWith(atask->getEndingTime());
            const long int xpos_start = static_cast<long  int>(double(hdimtime)*taskEndTime/duration) + hmargin;

            std::set<SpAbstractTask*> alreadyExist;

            for(const auto& taskDep : deps){
                if(alreadyExist.find(taskDep) == alreadyExist.end()){
                    const long int ypos_end = (threadIdsToVerticalSlotPosMap.at(taskDep->getThreadIdComputer())-1)*(vsizeperthread+threadstrock) + threadstrock/2 + vmargin + vsizeperthread/2;
                    const long int depstrocke = 1;
                    const double taskStartTime = startingTime.differenceWith(taskDep->getStartingTime());
                    const long int xpos_end = static_cast<long  int>(double(hdimtime)*taskStartTime/duration) + hmargin;

                    svgfile << "  <line x1=\"" << xpos_start  << "\" y1=\"" << ypos_start
                            << "\" x2=\"" << xpos_end << "\" y2=\"" << ypos_end << "\" stroke=\"" << "gray" <<"\" stroke-width=\"" << depstrocke <<"\"  marker-end=\"url(#arrow)\" />\n";
                }
            }
            deps.clear();
        }
    }

    const long int offsetStat = nbThreads*(vsizeperthread+threadstrock) + vmarginthreadstats;
    const char* statsNames[] = {"Submited", "Ready"};
    for(int idxStat = 0 ; idxStat < 2 ; ++idxStat){
        svgfile << "  <rect width=\"" << hdimtime << "\" height=\"" << vsizeperthread
                << "\" x=\"" << hmargin << "\" y=\"" << offsetStat + idxStat*(vsizeperthread+threadstrock) + vmargin << "\" style=\"fill:white;stroke:black;stroke-width:" << threadstrock << "\" />\n";

        const std::string label = statsNames[idxStat];

        svgfile << "<text x=\"" << hmargin/2 << "\" y=\"" << offsetStat + idxStat*(vsizeperthread+threadstrock+50) + vmargin + label.size()*30 - vsizeperthread/2 <<
                   "\" font-size=\"30\" fill=\"black\" transform=\"rotate(-90, " << hmargin/2 << " "
                << offsetStat + idxStat*(vsizeperthread+threadstrock+50) + vmargin + label.size()*30 - vsizeperthread/2 << ")\">" << label << "</text>\n";
    }

    small_vector<int> nbReady(hdimtime, 0);
    small_vector<int> nbSubmited(hdimtime, 0);

    for(const auto& atask : tasksFinished){
        const double taskSubmitedTime = startingTime.differenceWith(atask->getCreationTime());
        const double taskReadyTime = startingTime.differenceWith(atask->getReadyTime());
        const double taskStartTime = startingTime.differenceWith(atask->getStartingTime());
        const long int xpos_submited = static_cast<long  int>(double(hdimtime)*taskSubmitedTime/duration);
        const long int xpos_ready = static_cast<long  int>(double(hdimtime)*taskReadyTime/duration);
        const long int xpos_start = static_cast<long  int>(double(hdimtime)*taskStartTime/duration);

        nbSubmited[xpos_submited] += 1;

        nbReady[xpos_ready] += 1;
        if(xpos_ready != xpos_start || xpos_ready == hdimtime-1){
            nbReady[xpos_start] -= 1;
        }
        else{
            nbReady[xpos_start+1] -= 1;
        }
    }

    int maxReady = 0;
    {
        int currentStat = 0;
        for(int idx = 0 ; idx < hdimtime ; ++idx){
            currentStat += nbReady[idx];
            maxReady = std::max(maxReady , currentStat);
        }
    }

    const std::reference_wrapper<const small_vector<int>> statVal[] = {nbSubmited, nbReady};
    const int maxStatVal[2] = {static_cast<int>(tasksFinished.size()), maxReady};

    for(int idxStat = 0 ; idxStat < 2 ; ++idxStat){
         svgfile << "<polyline points=\"";
         //20,20 40,25 60,40 80,120 120,140 200,180"
         const int maxStat = maxStatVal[idxStat];
         int currentStat = 0;
         for(int idx = 0 ; idx < hdimtime ; ++idx){
             currentStat += statVal[idxStat].get()[idx];
             const long int xpos = hmargin+idx;
             const long int ypos = offsetStat + idxStat*(vsizeperthread+threadstrock) + vmargin
                                   + vsizeperthread
                                   - static_cast<long int>(double(vsizeperthread-threadstrock)*double(currentStat)/double(maxStat))
                                   - threadstrock/2;
             svgfile << xpos << "," << ypos << " ";
         }
         svgfile << "\" style=\"fill:none;stroke:rgb(112,0,0);stroke-width:3\" />\n";
    }

    svgfile << "<text x=\"" << hmargin + hdimtime + 10
            << "\" y=\"" << offsetStat + 0*(vsizeperthread+threadstrock) + vmargin + 15 <<
               "\" font-size=\"30\" fill=\"black\">" << tasksFinished.size() << "</text>\n";
    svgfile << "<text x=\"" << hmargin + hdimtime + 10
            << "\" y=\"" << offsetStat + 1*(vsizeperthread+threadstrock) + vmargin + 15 <<
               "\" font-size=\"30\" fill=\"black\">" << maxReady << "</text>\n";

    svgfile << "</svg>\n";
}

}


#endif
