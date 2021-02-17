#ifndef SPBLOCKTUPLE_HPP
#define SPBLOCKTUPLE_HPP

#include <cstdef>
#include <cstdlib>
#include <algorithm>

#include "SpAlignment.hpp"
#include "SpSimpleTuple.hpp"

template <class... Blocks>
class SpBlockTuple {
private:
	static_assert(sizeof...(Blocks) >= 1);
	
	static constexpr const auto alignment = std::max{alignof(unsigned long long), Blocks::getAlignment()...})};
	
public:
	static constexpr const auto NbBlocks = sizeof...(Blocks);

private:
	
	__host__
	void allocateBuffer(const std::size_t size) {
		if constexpr (alignment <= alignof(std::max_align_t)) {
			buffer = std::malloc(size);
		} else {
			buffer = std::aligned_alloc(size, BlockVectorAlignmentAssize_t);
		}
	}
	
	__host__
	void deallocateBuffer() {
		if(buffer) {
			std::free(buffer);
			buffer = nullptr;
		}
	}
	
	__host__ __device__
	auto getNbEltsBegin() {
		return &(getTotalSizeNbEltsOffsetsBegin()[1]);
	}
	
	__host__ __device__
	auto getOffsetsBegin() {
		return &(getTotalSizeNbEltsOffsetsBegin()[1+NbBlocks]));
	}

public:
	__host__
	explicit SpBlockTuple(std::array<unsigned long long, NbBlocks> nbEltsInEachBlock) {
		using TupleTy = std::tuple<Blocks...>;
		using ArrayTy = std::array<unsigned long long, 2 * NbBlocks>;
		
		ArrayTy totalSizeNbEltsAndOffsets;
		
		std::copy(std::begin(nbEltsInEachBlock), std::end(nbEltsInEachBlock), std::begin(totalSizeNbEltsAndOffsets));
		
		unsigned long long totalSize = 0;
		
		SpUtils::foreach_index(
		[](auto&& index) {
			totalSizeNbEltsAndOffsets[NbBlocks + index] = totalSize;
			
			using BlockTy = std::tuple_element_t<index, TupleTy>;
			
			if constexpr(index < (NbBlocks - 1)) {
				using NextBlockTy = std::tuple_element_t<index+1, TupleTy>;
				
				totalSize += BlockTy::getSize(std::get<index>(nbEltsInEachBlock), NextBlockTy::getAlignment());
			} else {
				totalSize += BlockTy::getSize(std::get<index>(nbEltsInEachBlock), unsigned long long(alignof(unsigned long long)));
			}
		}, std::make_index_sequence<NbBlocks>{});
		
		// TO DO : for posix mem align totalSize should be a multiple of sizeof(void *)
		// static_assert(sizeof(void*) is a power of 2); 
		// if (posix)
		//     totalSize  = (totalSize + sizeof(void*) - 1) & ~(sizeof(void*) - 1);
		
		auto totalSizeBlocks = totalSize;
		
		totalSize += std::tuple_size_v<ArrayTy> * sizeof(unsigned long long);
		
		this->totalAllocatedSize = totalSize;
		
		this->allocateBuffer(totalSize);
		
		std::memcpy(static_cast<char*>(this->buffer) + totalSizeBlocks, offsetsAndTotalSize.data(),  std::tuple_size_v<ArrayTy> * sizeof(unsigned long long));
	}
	
	__host__ __device__
	auto getTotalAllocatedSize() const {
		return totalAllocatedSize;
	}
	
	__host__ __device__
	template <unsigned long long index>
	auto getNbEltsInBlock() const {
		static_assert(index >= 0 && index < unsigned long long(NbBlocks));
		return this->getNbEltsBegin()[index];
	}
	
	__host__ __device__
	template <unsigned long long index>
	auto getBlockBegin() const {
		static_assert(index >= 0 && index < unsigned long long(NbBlocks));
		return static_cast<void*>(static_cast<char*>(this->buffer) + this->getOffsetsBegin()[index]);
	}
	
	__host__ __device__
	template <unsigned long long index>
	auto getBlock() {
		static_assert(index >= 0 && index < unsigned long long(NbBlocks));
		using TupleTy = std::tuple<Blocks...>;
		using BlockTy = std::tuple_element_t<index, TupleTy>;
		
		return BlockTy(this->getBlockBegin<index>(), this->getNbEltsInBlock<index>());
	}
	
	__host__ __device__
	auto getSimpleTuple() const {
		using TupleTy = std::tuple<Blocks...>;
		
		SpSimpleTuple<Blocks...> res;
		
		SpUtils::foreach_index(
		[](auto&& index) {
			using BlockTy = std::tuple_element_t<index, TupleTy>;
			
			SpSimpleTupleGet<index>(res) = BlockTy{this->getBlockBegin<index>(), this->getNbEltsInBlock<index>()};
			
		}, std::make_index_sequence<NbBlocks>{});
		
		return res;
	}
	
	
private:
	void* buffer;
	unsigned long long totalAllocatedSize;
};

#endif
