#include "SpCudaMemManager.hpp"

std::vector<SpCudaManager::SpCudaMemManager> SpCudaManager::Managers = SpCudaManager::BuildManagers();
