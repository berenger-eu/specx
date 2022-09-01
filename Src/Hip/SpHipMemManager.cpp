#include "SpHipMemManager.hpp"

std::vector<SpHipManager::SpHipMemManager> SpHipManager::Managers = SpHipManager::BuildManagers();
std::mutex SpHipManager::HipMutex;
