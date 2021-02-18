#ifndef SPBLOCKTUPLE_HPP
#define SPBLOCKTUPLE_HPP

#include <cstddef>
#include <cstdlib>
#include <algorithm>

#include "SpAlignment.hpp"

template <class... Blocks>
class SpBlockTuple {
private:
	static_assert(sizeof...(Blocks) >= 1);
	
	static constexpr const auto alignment = std::max({alignof(std::size_t), Blocks::getAlignment()...});
	
public:
	static constexpr const auto NbBlocks = sizeof...(Blocks);

private:
	
	void allocateBuffer(const std::size_t size) {
		if constexpr (alignment <= alignof(std::max_align_t)) {
			buffer = std::malloc(size);
		} else {
			buffer = std::aligned_alloc(size, alignment);
		}
	}
	
	void deallocateBuffer() {
		if(buffer) {
			std::free(buffer);
			buffer = nullptr;
		}
	}
	
	//__host__ __device__
	template <std::size_t index>
	auto getBlockBegin() const {
		static_assert(index < NbBlocks);
		return static_cast<void*>(static_cast<char*>(this->buffer) + this->getTotalAllocatedSize() - NbBlocks * sizeof(std::size_t));
	}
	
	//__host__ __device__
	template <std::size_t index>
	auto getNbEltsInBlock() const {
		static_assert(index < NbBlocks);
		return reinterpret_cast<std::size_t*>(static_cast<char*>(buffer) + this->getTotalAllocatedSize() - NbBlocks * 2 * sizeof(std::size_t));
	}

public:
	//__host__
	explicit SpBlockTuple(std::array<std::size_t, NbBlocks> nbEltsInEachBlock) {
		using TupleTy = std::tuple<Blocks...>;
		using ArrayTy = std::array<std::size_t, 2 * NbBlocks>;
		
		ArrayTy totalSizeNbEltsAndOffsets;
		
		std::copy(std::begin(nbEltsInEachBlock), std::end(nbEltsInEachBlock), std::begin(totalSizeNbEltsAndOffsets));
		
		std::size_t totalSize = 0;
		
		SpUtils::foreach_index(
		[&](auto&& index) {
			totalSizeNbEltsAndOffsets[NbBlocks + index] = totalSize;
			
			using BlockTy = std::tuple_element_t<index, TupleTy>;
			
			if constexpr(index < (NbBlocks - 1)) {
				using NextBlockTy = std::tuple_element_t<index+1, TupleTy>;
				
				totalSize += BlockTy::getSize(std::get<index>(nbEltsInEachBlock), NextBlockTy::getAlignment());
			} else {
				totalSize += BlockTy::getSize(std::get<index>(nbEltsInEachBlock), alignof(std::size_t));
			}
		}, std::make_index_sequence<NbBlocks>{});
		
		// TO DO : for posix mem align totalSize should be a multiple of sizeof(void *)
		// static_assert(sizeof(void*) is a power of 2); 
		// if (posix)
		//     totalSize  = (totalSize + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
		
		auto totalSizeBlocks = totalSize;
		
		totalSize += std::tuple_size_v<ArrayTy> * sizeof(std::size_t);
		
		this->totalAllocatedSize = totalSize;
		
		this->allocateBuffer(totalSize);
		
		std::memcpy(static_cast<char*>(this->buffer) + totalSizeBlocks, totalSizeNbEltsAndOffsets.data(),  std::tuple_size_v<ArrayTy> * sizeof(std::size_t));
	}
	
	SpBlockTuple(const SpBlockTuple& other) = default;
	SpBlockTuple(SpBlockTuple&& other) = default;
	SpBlockTuple& operator=(const SpBlockTuple& other) = default;
	SpBlockTuple& operator=(SpBlockTuple&& other) = default;
	/*
	//__device__
	explicit SpBlockTuple(std::pair<void*, std::size_t> p) : buffer(std::get<0>(p)), totalAllocatedSize(std::get<1>(p)) {}
	
	//__device__
	SpBlockTuple(const SpBlockTuple& other) = delete;
	//__device__
	SpBlockTuple(SpBlockTuple&& other) = delete;
	//__device__
	SpBlockTuple& operator=(const SpBlockTuple& other) = delete;
	//__device__
	SpBlockTuple& operator=(SpBlockTuple&& other) = delete;*/
	
	//__host__ __device__
	auto getTotalAllocatedSize() const {
		return totalAllocatedSize;
	}
	
	//__host__ __device__
	template <std::size_t index>
	auto getBlock() {
		static_assert(index < NbBlocks);
		using TupleTy = std::tuple<Blocks...>;
		using BlockTy = std::tuple_element_t<index, TupleTy>;
		
		return BlockTy(this->getBlockBegin<index>(), this->getNbEltsInBlock<index>());
	}
	
	//__host__ __device__
	auto getTuple() const {
		using TupleTy = std::tuple<Blocks...>;
		
		std::tuple<Blocks...> res;
		
		SpUtils::foreach_index(
		[&](auto&& index) {
			using BlockTy = std::tuple_element_t<index, TupleTy>;
			
			std::get<index>(res) = BlockTy{this->getBlockBegin<index>(), this->getNbEltsInBlock<index>()};
			
		}, std::make_index_sequence<NbBlocks>{});
		
		return res;
	}
	
	
private:
	void* buffer;
	unsigned long long totalAllocatedSize;
};

#endif
