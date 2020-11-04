#ifndef SPSECTTASKGROUP_HPP
#define SPSECTTASKGROUP_HPP

#include <vector>
#include <set>
#include <queue>
#include <algorithm>
#include <cassert>

#include "Task/SpAbstractTask.hpp"
#include "Speculation/SpSpeculativeModel.hpp"
#include "Utils/small_vector.hpp"

template <SpSpeculativeModel SpecModel>
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

    small_vector<SpGeneralSpecGroup*> parentGroups;

    int counterParentResults;
    SpecResult parentSpeculationResults;
    SpecResult selfSpeculationResults;

    small_vector<SpGeneralSpecGroup*> subGroups;

    std::atomic<States> state;

    small_vector<SpAbstractTask*> copyTasks;
    SpAbstractTask* mainTask;
    SpAbstractTask* specTask;
    small_vector<SpAbstractTask*> selectTasks;

    SpProbability selfProbability;

    bool isSpeculatif;
    
    std::shared_ptr<std::atomic<size_t>> numberOfSpeculativeSiblingSpecGroupsCounter;

    //////////////////////////////////////////////////////////////////

    static void EnableAllTasks(const small_vector_base<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setEnabled(SpTaskActivation::ENABLE);
        }
    }

    static void DisableAllTasks(const small_vector_base<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setEnabled(SpTaskActivation::DISABLE);
        }
    }
    
    static void DisableTasksDelegate(const small_vector_base<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setEnabledDelegate(SpTaskActivation::DISABLE);
        }
    }

    static void DisableIfPossibleAllTasks(const small_vector_base<SpAbstractTask*>& inTasks){
        for(auto* ptr : inTasks){
            ptr->setDisabledIfNotOver();
        }
    }

    //////////////////////////////////////////////////////////////////

public:
    SpGeneralSpecGroup(const bool inIsSpeculatif, std::shared_ptr<std::atomic<size_t>>& inNumberOfSpeculativeSiblingSpecGroupsCounter) :
        counterParentResults(0),
        parentSpeculationResults(SpecResult::UNDEFINED),
        selfSpeculationResults(SpecResult::UNDEFINED),
        state(States::UNDEFINED),
        mainTask(nullptr), specTask(nullptr),
        isSpeculatif(inIsSpeculatif),
        numberOfSpeculativeSiblingSpecGroupsCounter(inNumberOfSpeculativeSiblingSpecGroupsCounter){
    }
    
    virtual ~SpGeneralSpecGroup(){
        assert(isSpeculationDisable() || (didParentSpeculationSucceed() && counterParentResults == int(parentGroups.size())) 
                || !didParentSpeculationSucceed());
    }

    /////////////////////////////////////////////////////////////////////////

    void setProbability(const SpProbability& inSelfProbability){
        selfProbability = inSelfProbability;
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

            proba.append(currentGroup->selfProbability);

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
        
        if(inIsSpeculationEnable){
            state = States::DO_SPEC;
            if(isSpeculationResultUndefined()) {
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
        } else {
            state = States::DO_NOT_SPEC;
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

    /////////////////////////////////////////////////////////////////////////

    void addSubGroup(SpGeneralSpecGroup* inGroup){
        assert(std::find(subGroups.begin(), subGroups.end(), inGroup) ==  subGroups.end());
        assert(didSpeculationFailed() == false);
        subGroups.push_back(inGroup);
    }

    void addParents(small_vector_base<SpGeneralSpecGroup*> &inParents){
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
            for([[maybe_unused]] SpGeneralSpecGroup* gp : parentGroups){
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
        assert((isParentSpeculationResultUndefined() && counterParentResults < int(parentGroups.size()))
                || didParentSpeculationFailed());
        counterParentResults += 1;

        if(didParentSpeculationFailed()){
            // We know already that parents failed
        }
        else if(inSpeculationSucceed == false){
            // It is new, now we know parents failed
            parentSpeculationResults = SpecResult::SPECULATION_FAILED;
            // Inform children
            for(auto* child : subGroups){
                child->setParentSpecResult(false);
            }
            assert(mainTask->isEnable() == false);
            assert(specTask->isEnable());
            tryToEnableMainTask();
            specTask->setDisabledIfNotOver();
            DisableAllTasks(selectTasks);
        }
        else if(counterParentResults == int(parentGroups.size())){
            // All parents are over, and none of them failed, then it is a success!
            parentSpeculationResults = SpecResult::SPECULATION_SUCCED;
            assert(mainTask->isEnable() == false);
            assert(specTask->isEnable());
            if constexpr(SpecModel == SpSpeculativeModel::SP_MODEL_3) {
                if(*numberOfSpeculativeSiblingSpecGroupsCounter == 1) {
                    mainTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>()->setSpeculationCurrentResult(false);
                }
            }
            if(didSpeculationSucceed()){
                DisableTasksDelegate(selectTasks);

                for(auto* child : subGroups){
                    child->setParentSpecResult(true);
                }
                if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                    if(*numberOfSpeculativeSiblingSpecGroupsCounter == 1) {
                        mainTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>()->setSpeculationCurrentResult(true);
                    }
                }
            }
            else if(didSpeculationFailed()){
                // Already done EnableAllTasks(selectTasks);
                for(auto* child : subGroups){
                    child->setParentSpecResult(false);
                }
                if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                    if(*numberOfSpeculativeSiblingSpecGroupsCounter == 1) {
                        mainTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>()->setSpeculationCurrentResult(false);
                    }
                }
            }
        }
    }

    void setSpeculationCurrentResult(const bool inSpeculationSucceed){
        assert(isSpeculationEnable());
        assert((specTask != nullptr &&  parentGroups.size())
               || (specTask == nullptr &&  parentGroups.empty()));
        
        if(inSpeculationSucceed){
            selfSpeculationResults = SpecResult::SPECULATION_SUCCED;
        }
        else{
            selfSpeculationResults = SpecResult::SPECULATION_FAILED;
        }

        if(parentGroups.empty()){
            assert(specTask == nullptr);
            assert(selectTasks.size() == 0);

            for(auto* child : subGroups){
                child->setParentSpecResult(inSpeculationSucceed);
            }
        }
        else{
            assert(isSpeculatif);
            assert(specTask != nullptr);

            if(didParentSpeculationSucceed()){
                
                if constexpr(SpecModel != SpSpeculativeModel::SP_MODEL_3) {
                    if(*numberOfSpeculativeSiblingSpecGroupsCounter == 1) {
                        mainTask->getSpecGroup<SpGeneralSpecGroup<SpecModel>>()->setSpeculationCurrentResult(inSpeculationSucceed);
                    }
                }
                
                for(auto* child : subGroups){
                    child->setParentSpecResult(inSpeculationSucceed);
                }
                assert(mainTask->isEnable() == false);
                if(inSpeculationSucceed){
                    DisableTasksDelegate(selectTasks);
                }
            }
            else if(didParentSpeculationFailed()){
                // Check
                for([[maybe_unused]] auto* child : subGroups){
                    assert(child->didParentSpeculationFailed());
                }
            }else if(!inSpeculationSucceed) { // if current spec group failed,
                for(auto* child : subGroups){ // make all children fail right away 
                    child->setParentSpecResult(false);
                }
            }
            // If parentResult is undefined and current spec group succeeded, we have to wait
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
    
    SpTaskActivation getActivationStateForCopyTasks() {
        assert(didParentSpeculationFailed() == false);
        assert(didSpeculationFailed() == false);
        if(isSpeculationEnable()){
            return SpTaskActivation::ENABLE;
        }
        else {
            return SpTaskActivation::DISABLE;
        }
    }
    
    SpTaskActivation getActivationStateForMainTask() {
        if(!isSpeculatif || isSpeculationDisable() || didParentSpeculationFailed()){
            return SpTaskActivation::ENABLE;
        }
        else {
            return SpTaskActivation::DISABLE;
        }
    }
    
    SpTaskActivation getActivationStateForSpeculativeTask() {
        assert(didSpeculationFailed() == false);
        assert(isSpeculatif == true);
        
        if(isSpeculationEnable() && !didParentSpeculationFailed()){
            return SpTaskActivation::ENABLE;
        }
        else {
            return SpTaskActivation::DISABLE;
        }
    }
    
    SpTaskActivation getActivationStateForSelectTask(bool isCarryingSurelyWrittenValuesOver) {
        assert(mainTask);
        assert(specTask);
        assert(isSpeculatif == true);
        
        if(!isSpeculationEnable() || didParentSpeculationFailed()){
            return SpTaskActivation::DISABLE;
        }else if(isSpeculationEnable() && didParentSpeculationSucceed() && didSpeculationSucceed()){
            if(isCarryingSurelyWrittenValuesOver){
                return SpTaskActivation::ENABLE;
            } else {
                return SpTaskActivation::DISABLE;
            }
        }else{
            return SpTaskActivation::ENABLE;
        }
    }
    
    void addCopyTask(SpAbstractTask* inPreTask){
        assert(didParentSpeculationFailed() == false);
        assert(didSpeculationFailed() == false);
        assert(inPreTask->isOver() == false);
        copyTasks.push_back(inPreTask);
    }
    
    void addCopyTasks(const small_vector_base<SpAbstractTask*>& incopyTasks){
        copyTasks.reserve(copyTasks.size() + incopyTasks.size());
        copyTasks.insert(std::end(copyTasks), std::begin(incopyTasks), std::end(incopyTasks));
    }

    void setMainTask(SpAbstractTask* inMainTask){
        assert(mainTask == nullptr);
        mainTask = inMainTask;
    }

    void setSpecTask(SpAbstractTask* inSpecTask){
        assert(specTask == nullptr);
        specTask = inSpecTask;
    }

    void addSelectTasks(const small_vector_base<SpAbstractTask*>& inselectTasks){
        selectTasks.reserve(selectTasks.size() + inselectTasks.size());
        selectTasks.insert(std::end(selectTasks), std::begin(inselectTasks), std::end(inselectTasks));
    }
    
    void addSelectTask(SpAbstractTask* inSelectTask) {
        selectTasks.push_back(inSelectTask);
    }
    
    void tryToEnableMainTask() {
        (*numberOfSpeculativeSiblingSpecGroupsCounter)--;
        if(*numberOfSpeculativeSiblingSpecGroupsCounter == 0){
            mainTask->setEnabled(SpTaskActivation::ENABLE);
        }
    }
};



#endif


