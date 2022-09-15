#ifndef SPMAP_HPP
#define SPMAP_HPP

#include <vector>
#include <functional>

#include "small_vector.hpp"

template <class KeyType, class ValueType, std::size_t BucketSize = 128>
class SpMap{
    class Bucket{
        struct KKV{
            std::size_t hkey;
            KeyType key;
            ValueType value;

            KKV() = default;
            KKV(const KKV&) = default;
            KKV(KKV&&) = default;
            KKV& operator=(const KKV&) = default;
            KKV& operator=(KKV&&) = default;
        };

        small_vector<KKV> data;
        std::size_t nbItems;

    public:
        Bucket() : nbItems(0){}

        bool addOrReplace(const std::size_t hkey, KeyType&& inKey, ValueType&& inValue){
            for(auto& kkv : data){
                if(kkv.hkey == hkey && kkv.key == inKey){
                    kkv.value = std::move(inValue);
                    return false;
                }
            }
            data.emplace_back(KKV{hkey, std::move(inKey), std::move(inValue)});
            nbItems += 1;
            return true;
        }

        void add(const std::size_t hkey, KeyType&& inKey, ValueType&& inValue){
            assert(!exist(hkey, inKey));
            data.emplace_back(KKV{hkey, std::move(inKey), std::move(inValue)});
            nbItems += 1;
        }

        bool exist(const std::size_t hkey, const KeyType& inKey){
            for(const auto& kkv : data){
                if(kkv.hkey == hkey && kkv.key == inKey){
                    return true;
                }
            }
            return false;
        }

        std::size_t getSize() const{
            return nbItems;
        }

        bool remove(const std::size_t hkey, const KeyType& inKey){
            for(std::size_t idx = 0 ; idx < data.size() ; ++idx){
                const auto& kkv = data[idx];
                if(kkv.hkey == hkey && kkv.key == inKey){
                    if(idx != data.size() - 1){
                        data[idx] = std::move(data.back());
                    }
                    data.resize(data.size()-1);
                    return true;
                }
            }
            return false;
        }

        std::optional<std::reference_wrapper<ValueType>> find(const std::size_t hkey, const KeyType& inKey){
            for(auto& kkv : data){
                if(kkv.hkey == hkey && kkv.key == inKey){
                    return std::optional<std::reference_wrapper<ValueType>>{kkv.value};
                }
            }
            return std::nullopt;
        }

        std::optional<std::reference_wrapper<const ValueType>> find(const std::size_t hkey, const KeyType& inKey) const{
            for(const auto& kkv : data){
                if(kkv.hkey == hkey && kkv.key == inKey){
                    return std::optional<std::reference_wrapper<const ValueType>>{kkv.value};
                }
            }
            return std::nullopt;
        }
    };

    Bucket buckets[BucketSize];
    std::hash<KeyType> hasher;
    std::size_t nbItems;

public:
    using KeyType_t = KeyType;
    using ValueType_t = ValueType;

    SpMap() : nbItems(0){
    }

    void addOrReplace(KeyType inKey, ValueType inValue){
        const std::size_t hkey = hasher(inKey);
        if(buckets[hkey%BucketSize].addOrReplace(hkey, std::move(inKey), std::move(inValue))){
            nbItems += 1;
        }
    }

    void add(KeyType inKey, ValueType inValue){
        const std::size_t hkey = hasher(inKey);
        buckets[hkey%BucketSize].add(hkey, std::move(inKey), std::move(inValue));
        nbItems += 1;
    }

    bool exist(const KeyType& inKey){
        const std::size_t hkey = hasher(inKey);
        return buckets[hkey%BucketSize].exist(hkey, inKey);
    }

    std::size_t getSize() const{
        return nbItems;
    }

    std::size_t size() const{
        return getSize();
    }

    bool remove(const KeyType& inKey){
        const std::size_t hkey = hasher(inKey);
        const bool hasBeenRemoved = buckets[hkey%BucketSize].remove(hkey, inKey);
        if(hasBeenRemoved){
            nbItems -= 1;
            return true;
        }
        return false;
    }

    auto find(const KeyType& inKey){
        const std::size_t hkey = hasher(inKey);
        return buckets[hkey%BucketSize].find(hkey, inKey);
    }

    auto find(const KeyType& inKey) const{
        const std::size_t hkey = hasher(inKey);
        return buckets[hkey%BucketSize].find(hkey, inKey);
    }
};


#endif
