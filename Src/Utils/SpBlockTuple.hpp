#ifndef SPBLOCKTUPLE_HPP
#define SPBLOCKTUPLE_HPP

#include <cstddef>
#include <cstdlib>
#include <algorithm>

#include "SpAlignment.hpp"
#include "Config/SpConfig.hpp"

template <class... Blocks>
class SpBlockTuple {
private:
	static_assert(sizeof...(Blocks) >= 1);
	
	static constexpr auto alignment = std::max({alignof(std::size_t), Blocks::getAlignment()...});
	
public:
	static constexpr auto NbBlocks = sizeof...(Blocks);

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
	
	template <std::size_t index>
	SPHOST SPDEVICE
	auto getBlockBegin() const {
		static_assert(index < NbBlocks);
		return static_cast<void*>(static_cast<char*>(this->buffer) + this->getTotalAllocatedSize() - NbBlocks * sizeof(std::size_t));
	}
	
	template <std::size_t index>
	SPHOST SPDEVICE
	auto getNbEltsInBlock() const {
		static_assert(index < NbBlocks);
		return *reinterpret_cast<std::size_t*>(static_cast<char*>(buffer) + this->getTotalAllocatedSize() - NbBlocks * 2 * sizeof(std::size_t));
	}

public:
	SPHOST
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
	
	#ifndef SPETABARU_COMPILE_WITH_CUDA
	
	SpBlockTuple(const SpBlockTuple& other) {
		*this = other;
	}
	
	SpBlockTuple(SpBlockTuple&& other) {
		*this = std::move(other);
	}
	
	SpBlockTuple& operator=(const SpBlockTuple& other) {
		if(this == std::addressof(other)) {
			return *this;
		}
		deallocateBuffer();
		allocateBuffer(other.totalAllocatedSize);
		totalAllocatedSize = other.totalAllocatedSize;
	}
	
	SpBlockTuple& operator=(SpBlockTuple&& other) {
		if(this == std::addressof(other)) {
			return *this;
		}
		deallocateBuffer();
		buffer = other.buffer;
		totalAllocatedSize = other.totalAllocatedSize;
		other.buffer = nullptr;
		other.totalAllocatedSize = 0;
	}
	
	~SpBlockTuple() {
		deallocateBuffer();
	}
	
	#else
	
	SPDEVICE
	explicit SpBlockTuple(std::pair<void* const, std::size_t> p) : buffer(std::get<0>(p)), totalAllocatedSize(std::get<1>(p)) {}
	
	SPDEVICE
	SpBlockTuple(const SpBlockTuple& other) = delete;
	
	SPDEVICE
	SpBlockTuple(SpBlockTuple&& other) = delete;
	
	SPDEVICE
	SpBlockTuple& operator=(const SpBlockTuple& other) = delete;
	
	SPDEVICE
	SpBlockTuple& operator=(SpBlockTuple&& other) = delete;
	
	#endif
	
	SPHOST SPDEVICE
	auto getTotalAllocatedSize() const {
		return totalAllocatedSize;
	}
	
	template <std::size_t index>
	SPHOST SPDEVICE
	auto getBlock() {
		static_assert(index < NbBlocks);
		using TupleTy = std::tuple<Blocks...>;
		using BlockTy = std::tuple_element_t<index, TupleTy>;
		
		return BlockTy(this->getBlockBegin<index>(), this->getNbEltsInBlock<index>());
	}
	
	SPHOST SPDEVICE
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
	std::size_t totalAllocatedSize;
};

#endif
