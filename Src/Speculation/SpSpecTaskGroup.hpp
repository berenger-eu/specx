#ifndef SPSECTTASKGROUP_HPP
#define SPSECTTASKGROUP_HPP

#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <cassert>

#include "Tasks/SpAbstractTask.hpp"

class SpGeneralSpecGroup{
protected:
    enum class States{
        UNDEFINED,
        DO_NOT_SPEC,
        DO_SPEC
    };

    enum class SpecResult{
        UNDEFINED,
        SPECULATION_FAILED,
        SPECULATION_SUCCED
    };

    //////////////////////////////////////////////////////////////

    std::vector<SpGeneralSpecGroup*> parentGroups;

    int counterParentResults;
    SpecResult parentSpeculationResults;
    SpecResult selfSpeculationResults;

    std::vector<SpGeneralSpecGroup*> subGroups;

    std::atomic<States> state;

    std::vector<SpAbstractTask*> copyTasks;
    SpAbstractTask* mainTask;
    SpAbstractTask* specTask;
    std::vector<SpAbstractTask*> selectTasks;

    SpProbability selfPropability;

    bool isSpeculatif;

    //////////////////////////////////////////////////////////////////

    static void EnableAllTasks(const std::vector<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setEnabled(SpTaskActivation::ENABLE);
        }
    }

    static void DisableAllTasks(const std::vector<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setEnabled(SpTaskActivation::DISABLE);
        }
    }

    static void DisableIfPossibleAllTasks(const std::vector<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setDisabledIfNotOver();
        }
    }

    //////////////////////////////////////////////////////////////////

public:
    SpGeneralSpecGroup(const bool inIsSpeculatif) :
        counterParentResults(0),
        parentSpeculationResults(SpecResult::UNDEFINED),
        selfSpeculationResults(SpecResult::UNDEFINED),
        state(States::UNDEFINED),
        mainTask(nullptr), specTask(nullptr),
        isSpeculatif(inIsSpeculatif){
    }

    virtual ~SpGeneralSpecGroup(){
        assert(isSpeculationDisable() || counterParentResults == int(parentGroups.size()));
    }

    /////////////////////////////////////////////////////////////////////////

    void setProbability(const SpProbability& inSelfPropability){
        selfPropability = inSelfPropability;
    }

    SpProbability getAllProbability(){
        SpProbability proba;

        std::set<SpGeneralSpecGroup*> groupsIncluded;
        std::queue<SpGeneralSpecGroup*> toProceed;

        toProceed.push(this);
        groupsIncluded.insert(this);

        while(toProceed.size()){
            SpGeneralSpecGroup* currentGroup = toProceed.front();
            toProceed.pop();

            proba.append(currentGroup->selfPropability);

            for(auto* parent : parentGroups){
                if(groupsIncluded.find(parent) == groupsIncluded.end()){
                    toProceed.push(parent);
                    groupsIncluded.insert(parent);
                }
            }
            for(auto* child : subGroups){
                if(groupsIncluded.find(child) == groupsIncluded.end()){
                    toProceed.push(child);
                    groupsIncluded.insert(child);
                }
            }
        }

        return proba;
    }

    /////////////////////////////////////////////////////////////////////////

    void setSpeculationActivationCore(const bool inIsSpeculationEnable){
        assert(isSpeculationEnableOrDisable() == false);
        assert(isSpeculationResultUndefined() == true);

        if(inIsSpeculationEnable){
            state = States::DO_SPEC;

            if(mainTask){
                assert(mainTask->isOver() == false);
                if(isSpeculatif){
                    mainTask->setEnabled(SpTaskActivation::DISABLE);
                }
                else{
                    mainTask->setEnabled(SpTaskActivation::ENABLE);
                }
            }

            if(specTask){
                assert(isSpeculatif);
                assert(specTask->isOver() == false);
                specTask->setEnabled(SpTaskActivation::ENABLE);
            }

            EnableAllTasks(copyTasks);
            EnableAllTasks(selectTasks);
        }
    }

    void setSpeculationActivation(const bool inIsSpeculationEnable){
        assert(isSpeculationEnableOrDisable() == false);

        std::set<SpGeneralSpecGroup*> groupsIncluded;
        std::queue<SpGeneralSpecGroup*> toProceed;

        toProceed.push(this);
        groupsIncluded.insert(this);

        while(toProceed.size()){
            SpGeneralSpecGroup* currentGroup = toProceed.front();
            toProceed.pop();

            assert(currentGroup->isSpeculationEnableOrDisable() == false);
            currentGroup->setSpeculationActivationCore(inIsSpeculationEnable);
            assert(currentGroup->isSpeculationEnableOrDisable() == true);

            for(auto* parent : currentGroup->parentGroups){
                if(groupsIncluded.find(parent) == groupsIncluded.end()
                        && parent->isSpeculationEnableOrDisable() == false){
                    toProceed.push(parent);
                    groupsIncluded.insert(parent);
                }
            }
            for(auto* child : currentGroup->subGroups){
                if(groupsIncluded.find(child) == groupsIncluded.end()
                        && child->isSpeculationEnableOrDisable() == false){
                    toProceed.push(child);
                    groupsIncluded.insert(child);
                }
            }
        }
    }

    /////////////////////////////////////////////////////////////////////////

    bool isSpeculationEnableOrDisable() const {
        return state != States::UNDEFINED;
    }

    bool isSpeculationNotSet() const {
        return isSpeculationEnableOrDisable() == false;
    }

    bool isSpeculationEnable() const {
        return state == States::DO_SPEC;
    }

    bool isSpeculationDisable() const {
        return state == States::DO_NOT_SPEC;
    }
    
    SpAbstractTask* getSpecTask()
    {
      return specTask;
    }
    
    SpAbstractTask* getMainTask()
    {
      return mainTask;
    }

    /////////////////////////////////////////////////////////////////////////

    void addSubGroup(SpGeneralSpecGroup* inGroup){
        assert(std::find(subGroups.begin(), subGroups.end(), inGroup) ==  subGroups.end());
        assert(didSpeculationFailed() == false);
        subGroups.push_back(inGroup);
    }

    void addParents(std::vector<SpGeneralSpecGroup*> inParents){
        assert(parentGroups.empty());
        assert(isParentSpeculationResultUndefined());
        assert(isSpeculationResultUndefined());

        // Put parents into current group's parent list
        parentGroups = std::move(inParents);

        // Check if one parent has already speculation activated
        bool oneGroupSpecEnable = false;
        for(SpGeneralSpecGroup* gp : parentGroups){
            assert(gp->isSpeculationDisable() == false);
            if(gp->isSpeculationEnable()){
                oneGroupSpecEnable = true;
                break;
            }
        }
        // Yes, so activate all parents
        if(oneGroupSpecEnable){
            for(SpGeneralSpecGroup* gp : parentGroups){
                assert(gp->didSpeculationFailed() == false);
                assert(gp->didParentSpeculationFailed() == false);

                if(gp->isSpeculationNotSet()){
                    assert(gp->isSpeculationResultUndefined() == true);
                    assert(gp->isParentSpeculationResultUndefined() == true);

                    gp->setSpeculationActivation(true);
                }
                else if(gp->didAllSpeculationSucceed()){
                    counterParentResults += 1;
                }
            }

            setSpeculationActivationCore(true);

            // All results from parents are known
            if(counterParentResults == int(parentGroups.size())){
                parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
            }
        }

        // Put current group into parents' subGroup lists
        for(auto* ptr : parentGroups){
            ptr->addSubGroup(this);
        }

        // Simple check
        if(oneGroupSpecEnable){
            for(SpGeneralSpecGroup* gp : parentGroups){
                assert(gp->isSpeculationEnable());
            }
        }
        else{
            assert(counterParentResults == 0);
        }
    }

    /////////////////////////////////////////////////////////////////////////

    void setParentSpecResult(const bool inSpeculationSucceed){
        assert(isSpeculatif);
        assert(counterParentResults < int(parentGroups.size()));
        counterParentResults += 1;

        if(didParentSpeculationFailed()){
            // We know already that parents failed
        }
        else if(inSpeculationSucceed == false){
            // It is new, now we know parents failed
            parentSpeculationResults = SpecResult::SPECULATION_FAILED;
            // Inform children
//            for(auto* child : subGroups){
//                child->setParentSpecResult(false);
//            }
            assert(mainTask->isEnable() == false);
            assert(specTask->isEnable());
            mainTask->setEnabled(SpTaskActivation::ENABLE);
            specTask->setDisabledIfNotOver();
            DisableAllTasks(selectTasks);
        }
        else if(counterParentResults == int(parentGroups.size())){
            // All parents are over, and none of them failed, then it is a success!
            parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
            assert(mainTask->isEnable() == false);
            assert(specTask->isEnable());
            if(didSpeculationSucceed()){
                DisableAllTasks(selectTasks);
                for(auto* child : subGroups){
                    child->setParentSpecResult(true);
                }
            }
            else if(didSpeculationFailed()){
                // Already done EnableAllTasks(selectTasks);
//                for(auto* child : subGroups){
//                    child->setParentSpecResult(false);
//                }
            }
        }
    }

    void setSpeculationCurrentResult(const bool inSpeculationSucceed, const bool isEnable, const bool isOnNormalPath){
        assert(isSpeculatif == false || isOnNormalPath == false || didParentSpeculationFailed() == isEnable);
        if(didParentSpeculationFailed()){
            for(auto& slt : selectTasks){
                // simply ensure that selects are disable
                assert(slt->isEnable() == false);
            }
        }
        if(parentGroups.empty()){
            assert(specTask == nullptr);
            assert(selectTasks.size() == 0);
            assert(isOnNormalPath == true);
        }
        else{
            assert(isSpeculatif);
            assert(specTask != nullptr);
            assert(selectTasks.size() != 0);
        }
        assert(!isSpeculatif || mainTask->isEnable() == didParentSpeculationFailed());
        assert((specTask != nullptr &&  parentGroups.size())
               || (specTask == nullptr &&  parentGroups.empty()));

        if(isOnNormalPath){
            assert(isSpeculatif == false || didParentSpeculationSucceed() || didParentSpeculationFailed());
            if(isSpeculatif == false || didParentSpeculationFailed()){
                assert(isEnable);
                if(inSpeculationSucceed){
                    selfSpeculationResults = SpecResult::SPECULATION_SUCCED;
                }
                else{
                    selfSpeculationResults = SpecResult::SPECULATION_FAILED;
                }
                for(auto* child : subGroups){
                    child->setParentSpecResult(inSpeculationSucceed);
                }
            }
        }
        else{
            assert(isEnable);
            if(inSpeculationSucceed){
                selfSpeculationResults = SpecResult::SPECULATION_SUCCED;
            }
            else{
                selfSpeculationResults = SpecResult::SPECULATION_FAILED;
            }
            if(didParentSpeculationSucceed()){
                for(auto* child : subGroups){
                    child->setParentSpecResult(inSpeculationSucceed);
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    bool isSpeculationResultUndefined() const{
        return selfSpeculationResults == SpecResult::UNDEFINED;
    }

    bool isParentSpeculationResultUndefined() const{
        return parentSpeculationResults == SpecResult::UNDEFINED;
    }

    bool didSpeculationFailed() const{
        return selfSpeculationResults == SpecResult::SPECULATION_FAILED;
    }

    bool didParentSpeculationFailed() const{
        return parentSpeculationResults == SpecResult::SPECULATION_FAILED;
    }

    bool didSpeculationSucceed() const{
        return selfSpeculationResults == SpecResult::SPECULATION_SUCCED;
    }

    bool didParentSpeculationSucceed() const{
        return parentSpeculationResults == SpecResult::SPECULATION_SUCCED;
    }

    bool didAllSpeculationSucceed() const{
        return (parentGroups.empty() || parentSpeculationResults == SpecResult::SPECULATION_SUCCED)
                && selfSpeculationResults == SpecResult::SPECULATION_SUCCED;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    void addCopyTasks(const std::vector<SpAbstractTask*>& incopyTasks){
        assert(didParentSpeculationFailed() == false);
        assert(didSpeculationFailed() == false);

        if(isSpeculationEnable()){
            EnableAllTasks(incopyTasks);
        }
        else {
            DisableAllTasks(incopyTasks);
        }
        copyTasks.reserve(copyTasks.size() + incopyTasks.size());
        copyTasks.insert(std::end(copyTasks), std::begin(incopyTasks), std::end(incopyTasks));
    }

    void addCopyTask(SpAbstractTask* inPreTask){
        assert(didParentSpeculationFailed() == false);
        assert(didSpeculationFailed() == false);
        assert(inPreTask->isOver() == false);

        if(isSpeculationEnable()){
            inPreTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inPreTask->setEnabled(SpTaskActivation::DISABLE);
        }
        copyTasks.push_back(inPreTask);
    }

    void setMainTask(SpAbstractTask* inMainTask){
        if(isSpeculatif == false || isSpeculationDisable() || didParentSpeculationFailed()){
            inMainTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inMainTask->setEnabled(SpTaskActivation::DISABLE);
        }
        assert(mainTask == nullptr);
        mainTask = inMainTask;
    }

    void setSpecTask(SpAbstractTask* inSpecTask){
        assert(didSpeculationFailed() == false);
        assert(isSpeculatif == true);

        if(isSpeculationEnable() && didParentSpeculationFailed() == false){
            inSpecTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inSpecTask->setEnabled(SpTaskActivation::DISABLE);
        }
        assert(specTask == nullptr);
        specTask = inSpecTask;
    }

    void addSelectTasks(const std::vector<SpAbstractTask*>& inselectTasks){
        assert(mainTask);
        assert(specTask);
        assert(isSpeculatif == true);

        if(isSpeculationEnable() &&
                (didParentSpeculationFailed() == false
                || (didParentSpeculationSucceed() && didSpeculationSucceed()) == false)){
            EnableAllTasks(inselectTasks);
        }
        else{
            DisableAllTasks(inselectTasks);
        }
        selectTasks.reserve(selectTasks.size() + inselectTasks.size());
        selectTasks.insert(std::end(selectTasks), std::begin(inselectTasks), std::end(inselectTasks));
    }

    
    void disableOrEnableCopyTasks(bool enable)
    {
      if (enable) {
        EnableAllTasks(copyTasks);
      } else {
        DisableAllTasks(copyTasks);
      }
    }
    
};



#endif
