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

    const bool isUncertain;
    const bool isSpeculative;

    std::vector<SpGeneralSpecGroup*> parentGroups;
    SpecResult parentSpeculationResults;
    SpecResult selfSpeculationResults; // Only if uncertain
    int parentSpeculationCounter;

    std::vector<SpGeneralSpecGroup*> subGroupsNormalPath;
    std::vector<SpGeneralSpecGroup*> subGroupsSpeculativePath;

    std::atomic<States> state;

    //////////////////////////////////////////////////////////////

    std::atomic<bool> isGroupEnable;

    std::vector<SpAbstractTask*> preTasks;
    SpAbstractTask* mainTask;
    std::vector<SpAbstractTask*> postTasksIfSucceed; // Only if uncertain
    std::vector<SpAbstractTask*> postTasksIfFailed; // Only if uncertain

    SpProbability selfPropability;

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


    void setEnable(){
        if(isGroupEnable == true){
            return;
        }

        isGroupEnable = true;

        if(mainTask){
            mainTask->setEnabled(SpTaskActivation::ENABLE);
        }
        EnableAllTasks(preTasks);
        EnableAllTasks(postTasksIfSucceed);
    }

    void setEnableCurrentAndNormalChildParentFailed(){
        setEnable();
        parentSpeculationResults = SpecResult::SPECULATION_FAILED;
        for(auto* child : subGroupsNormalPath){
            assert(child->isSpeculative == false);
            child->setEnableCurrentAndNormalChildParentFailed();
        }
    }

    void setDisableCurrentAndNormalChildParentSucced(){
        //setDisable();
        assert(isGroupEnable == false);
        parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
        for(auto* child : subGroupsNormalPath){
            assert(child->isSpeculative == false);
            if(child->parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                child->setDisableCurrentAndNormalChildParentSucced();
            }
        }
    }

    void setEnableCurrentAndNormalChildNoSpec(){
        assert(state == States::DO_NOT_SPEC
               || state == States::UNDEFINED);
        setEnable();
        for(auto* child : subGroupsNormalPath){
            assert(child->isSpeculative == false);
            assert(child->state == States::DO_NOT_SPEC
                   || child->state == States::UNDEFINED);
            child->setEnableCurrentAndNormalChildNoSpec();
        }
    }

    void setDisableParentFailed(){
        assert(state == States::DO_SPEC);
        setDisable();
        for(auto* child : subGroupsSpeculativePath){
            child->setDisableParentFailed();
        }
    }

   void setDisable(){
       assert(isGroupEnable == true);

       isGroupEnable = false;

       if(mainTask){
           mainTask->setDisabledIfNotOver();
       }
       DisableIfPossibleAllTasks(preTasks);
       DisableAllTasks(postTasksIfSucceed);
       EnableAllTasks(postTasksIfFailed);
   }

public:
    SpGeneralSpecGroup(const bool inIsUncertain, const bool inIsSpeculative) :
        isUncertain(inIsUncertain), isSpeculative(inIsSpeculative), parentSpeculationResults(SpecResult::UNDEFINED),
        selfSpeculationResults(SpecResult::UNDEFINED),
         parentSpeculationCounter(0), state(States::UNDEFINED),
         isGroupEnable(false), mainTask(nullptr){
    }

    virtual ~SpGeneralSpecGroup(){}

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


            if(currentGroup->isSpeculative){
                proba.append(currentGroup->selfPropability);
            }

            for(auto* parent : parentGroups){
                if(groupsIncluded.find(parent) == groupsIncluded.end()){
                    toProceed.push(parent);
                    groupsIncluded.insert(parent);
                }
            }
            for(auto* child : subGroupsNormalPath){
                assert(child->isSpeculative == false);
                if(groupsIncluded.find(child) == groupsIncluded.end()){
                    toProceed.push(child);
                    groupsIncluded.insert(child);
                }
            }
            for(auto* child : subGroupsSpeculativePath){
                assert(child->isSpeculative == true);
                if(groupsIncluded.find(child) == groupsIncluded.end()){
                    toProceed.push(child);
                    groupsIncluded.insert(child);
                }
            }
        }

        return proba;
    }

    void setSpeculationActivation(const bool inIsSpeculationEnable, const bool propagate = true){
        assert(isSpeculationEnableOrDisable() == false);

        std::set<SpGeneralSpecGroup*> groupsIncluded;
        std::queue<SpGeneralSpecGroup*> toProceed;

        toProceed.push(this);
        groupsIncluded.insert(this);

        while(toProceed.size()){
            SpGeneralSpecGroup* currentGroup = toProceed.front();
            toProceed.pop();

            if(currentGroup->parentGroups.empty()){
                currentGroup->setEnable();
            }

            if(inIsSpeculationEnable){
                currentGroup->state = States::DO_SPEC;
                if(currentGroup->isSpeculative){
                    assert(!currentGroup->parentGroups.empty());
                    currentGroup->setEnable();
                }
            }
            else{
                currentGroup->state = States::DO_NOT_SPEC;
                if(currentGroup->parentGroups.empty()){
                    assert(currentGroup->isUncertain == true);
                    assert(currentGroup->isSpeculative == false);

                    for(auto* child : currentGroup->subGroupsNormalPath){
                        assert(child->isSpeculative == false);
                        child->setEnableCurrentAndNormalChildNoSpec();
                    }
                }
            }

            if(propagate){
                for(auto* parent : currentGroup->parentGroups){
                    if(groupsIncluded.find(parent) == groupsIncluded.end()){
                        assert(parent->isSpeculationEnableOrDisable() == false);
                        toProceed.push(parent);
                        groupsIncluded.insert(parent);
                    }
                }
                for(auto* child : currentGroup->subGroupsNormalPath){
                    assert(child->isSpeculative == false);
                    if(groupsIncluded.find(child) == groupsIncluded.end()){
                        assert(child->isSpeculationEnableOrDisable() == false);
                        toProceed.push(child);
                        groupsIncluded.insert(child);
                    }
                }
                for(auto* child : currentGroup->subGroupsSpeculativePath){
                    assert(child->isSpeculative == true);
                    if(groupsIncluded.find(child) == groupsIncluded.end()){
                        assert(child->isSpeculationEnableOrDisable() == false);
                        toProceed.push(child);
                        groupsIncluded.insert(child);
                    }
                }
            }
        }
    }

    bool isSpeculationEnableOrDisable() const {
        return state != States::UNDEFINED;
    }

    bool isSpeculationUndefined() const {
        return isSpeculationEnableOrDisable() == false;
    }

    bool isSpeculationEnable() const {
        return state == States::DO_SPEC;
    }

    bool isSpeculationDisable() const {
        return state == States::DO_NOT_SPEC;
    }

    bool isEnable(){
        return isGroupEnable;
    }

    bool isDisable(){
        return isGroupEnable == false;
    }


    void addGroupInNormalPath(SpGeneralSpecGroup* inGroup){
        assert(std::find(subGroupsNormalPath.begin(), subGroupsNormalPath.end(), inGroup) ==  subGroupsNormalPath.end());
        assert(isUncertain == true);
        assert(inGroup->isSpeculative == false);
        assert(didSpeculationFailed() == false);
        subGroupsNormalPath.push_back(inGroup);
    }

    void addGroupInSpeculativePath(SpGeneralSpecGroup* inGroup){
        assert(std::find(subGroupsSpeculativePath.begin(), subGroupsSpeculativePath.end(), inGroup) ==  subGroupsSpeculativePath.end());
        assert(isUncertain == true);
        assert(inGroup->isSpeculative == true);
        subGroupsSpeculativePath.push_back(inGroup);
    }

    void addThisGroupToParentsInNormalPath(const std::vector<SpGeneralSpecGroup*>& inParents){
        assert(parentGroups.empty());

        bool oneGroupSpecEnable = false;
        for(SpGeneralSpecGroup* gp : inParents){
            assert(gp->isSpeculationDisable() == false);
            if(gp->isSpeculationEnable()){
                oneGroupSpecEnable = true;
                break;
            }
        }
        if(oneGroupSpecEnable){
            for(SpGeneralSpecGroup* gp : inParents){
                if(gp->isSpeculationUndefined()){
                    gp->setSpeculationActivation(true);
                }
            }
        }
        for(auto* ptr : inParents){
            ptr->addGroupInNormalPath(this);
        }
        parentGroups = inParents;
        if(oneGroupSpecEnable){
            setSpeculationActivation(true, false);
            for(SpGeneralSpecGroup* gp : inParents){
                assert(gp->isSpeculationEnable());
            }
        }
    }

    void addThisGroupToParentsInSpeculativePath(const std::vector<SpGeneralSpecGroup*>& inParents){
        assert(parentGroups.empty());

        bool oneGroupSpecEnable = false;
        for(SpGeneralSpecGroup* gp : inParents){
            assert(gp->isSpeculationDisable() == false);
            if(gp->isSpeculationEnable()){
                oneGroupSpecEnable = true;
                break;
            }
        }
        if(oneGroupSpecEnable){
            for(SpGeneralSpecGroup* gp : inParents){
                if(gp->isSpeculationUndefined()){
                    gp->setSpeculationActivation(true);
                }
            }
        }
        for(auto* ptr : inParents){
            ptr->addGroupInSpeculativePath(this);
        }
        parentGroups = inParents;
        if(oneGroupSpecEnable){
            setSpeculationActivation(true, false);
            setEnable();
            for(SpGeneralSpecGroup* gp : inParents){
                assert(gp->isSpeculationEnable());
            }
        }
    }

    void setSpeculationCurrentResult(bool inSpeculationSucceed){
        assert(isSpeculationEnable());
        assert(isUncertain == true);
        assert(isEnable() == true);

        if(parentGroups.empty()){
            parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
        }

        if(inSpeculationSucceed){
            selfSpeculationResults = SpecResult::SPECULATION_SUCCED;
            if(parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                EnableAllTasks(postTasksIfSucceed);
                DisableAllTasks(postTasksIfFailed);

                for(auto* child : subGroupsNormalPath){
                    assert(child->isSpeculative == false);
                    if(child->parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                        assert(child->isGroupEnable == false);
                        assert(child->postTasksIfSucceed.empty());
                        assert(child->postTasksIfFailed.empty());
                        child->setDisableCurrentAndNormalChildParentSucced();
                    }
                }
                for(auto* child : subGroupsSpeculativePath){
                    assert(child->isSpeculative == true);
                    if(child->selfSpeculationResults  != SpecResult::SPECULATION_FAILED &&
                            child->parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                        assert(child->isGroupEnable == true);
                        child->parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
                        if(child->isUncertain == false){
                            EnableAllTasks(child->postTasksIfSucceed);
                            DisableAllTasks(child->postTasksIfFailed);
                        }
                    }
                }
            }
        }
        else{
            selfSpeculationResults = SpecResult::SPECULATION_FAILED;

            if(parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                EnableAllTasks(postTasksIfFailed);
                DisableAllTasks(postTasksIfSucceed);

                for(auto* child : subGroupsNormalPath){
                    assert(child->isSpeculative == false);
                    if(child->parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                        assert(child->isGroupEnable == false);
                        child->setEnableCurrentAndNormalChildParentFailed();
                        assert(child->postTasksIfSucceed.empty());
                        assert(child->postTasksIfFailed.empty());
                    }
                }
                for(auto* child : subGroupsSpeculativePath){
                    assert(child->isSpeculative == true);
                    if(child->selfSpeculationResults  != SpecResult::SPECULATION_FAILED &&
                            child->parentSpeculationResults != SpecResult::SPECULATION_FAILED){
                        assert(child->isGroupEnable == true);
                        child->parentSpeculationResults = SpecResult::SPECULATION_FAILED;
                        child->setDisableParentFailed();
                    }
                }
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////

    bool didSpeculationFailed() const{
        return selfSpeculationResults == SpecResult::SPECULATION_FAILED
                || parentSpeculationResults == SpecResult::SPECULATION_FAILED;
    }

    bool didSpeculationSucceed() const{
        return selfSpeculationResults == SpecResult::SPECULATION_SUCCED
                && parentSpeculationResults == SpecResult::SPECULATION_SUCCED;
    }

    void addPreTasks(const std::vector<SpAbstractTask*>& inPreTasks){
        if(isGroupEnable){
            EnableAllTasks(inPreTasks);
        }
        else {
            DisableAllTasks(inPreTasks);
        }
        preTasks.reserve(preTasks.size() + inPreTasks.size());
        preTasks.insert(std::end(preTasks), std::begin(inPreTasks), std::end(inPreTasks));
    }

    void addPreTask(SpAbstractTask* inPreTask){
        if(isGroupEnable){
            inPreTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inPreTask->setEnabled(SpTaskActivation::DISABLE);
        }
        preTasks.push_back(inPreTask);
    }

    void setMainTask(SpAbstractTask* inMainTask){
        if(isGroupEnable){
            inMainTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inMainTask->setEnabled(SpTaskActivation::DISABLE);
        }
        assert(mainTask == nullptr);
        mainTask = inMainTask;
    }

    void addPostTasks(const bool ifSucceed, const std::vector<SpAbstractTask*>& inPostTasks){
        if((didSpeculationSucceed() && ifSucceed)
               || (didSpeculationFailed() && !ifSucceed) ){
            EnableAllTasks(inPostTasks);
        }
        else {
            DisableAllTasks(inPostTasks);
        }
        if(ifSucceed){
            postTasksIfSucceed.reserve(postTasksIfSucceed.size() + inPostTasks.size());
            postTasksIfSucceed.insert(std::end(postTasksIfSucceed), std::begin(inPostTasks), std::end(inPostTasks));
        }
        else{
            postTasksIfFailed.reserve(postTasksIfFailed.size() + inPostTasks.size());
            postTasksIfFailed.insert(std::end(postTasksIfFailed), std::begin(inPostTasks), std::end(inPostTasks));
        }
    }

    void addPostTask(const bool ifSucceed, SpAbstractTask* inPostTask){
        if((didSpeculationSucceed() && ifSucceed)
               || (didSpeculationFailed() && !ifSucceed) ){
            inPostTask->setEnabled(SpTaskActivation::ENABLE);
        }
        else {
            inPostTask->setEnabled(SpTaskActivation::DISABLE);
        }
        if(ifSucceed){
            postTasksIfSucceed.push_back(inPostTask);
        }
        else{
            postTasksIfFailed.push_back(inPostTask);
        }
    }
};



#endif


