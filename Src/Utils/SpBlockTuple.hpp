#ifndef SPBLOCKTUPLE_HPP
#define SPBLOCKTUPLE_HPP

#include <cstdef>
#include <cstdlib>
#include <algorithm>

#include "SpAlignment.hpp"
#include "SpSimpleTuple.hpp"

template <class... Blocks, const SpAlignment BlockAlignment=SpAlignment{std::max{Blocks::getAlignment()...}}}>
class SpBlockTuple {
private:
	static_assert(sizeof...(Blocks) >= 1);
	static_assert(BlockAlignment.isPowerOf2());
	static_assert(SpAlignment{alignof(std::size_t)}.isPowerOf2());
	
	static constexpr const auto BlockAlignmentAssize_t = static_cast<std::size_t>(BlockAlignment);
	static constexpr const auto BlockVectorAlignmentAssize_t = static_cast<std::size_t>(SpAlignment{alignof(std::size_t, Blocks::value_type...)});
	
	static_assert(std::conjunction_v<std::bool_constant<BlockAlignmentAssize_t >= alignof(Blocks::value_type)>...>);
	static_assert(std::conjunction_v<std::is_trivially_copyable<Blocks::value_type>...>);
	
public:
	static constexpr const auto NbBlocks = sizeof...(Blocks);

private:
	
	void allocateBuffer(const std::size_t size) {
		if constexpr (BlockVectorAlignmentAssize_t <= alignof(std::max_align_t)) {
			buffer = std::malloc(size);
		} else {
			buffer = std::aligned_alloc(size, BlockVectorAlignmentAssize_t);
		}
	}
	
	void deallocateBuffer() {
		if(buffer) {
			std::free(buffer);
			buffer = nullptr;
		}
	}
	
	auto getTotalSizeNbEltsOffsetsBegin() {
		return static_cast<std::size_t*>(buffer);
	}
	
	auto getNbEltsBegin() {
		return std::addressof(getTotalSizeNbEltsOffsetsBegin()[1]);
	}
	
	auto getOffsetsBegin() {
		return std::addressof(getTotalSizeNbEltsOffsetsBegin()[1+NbBlocks]);
	}

public:
	
	explicit SpBlockTuple(std::array<std::size_t, NbBlocks> nbEltsInEachBlock) {
		using TupleTy = std::tuple<Blocks...>;
		using ArrayTy = std::array<std::size_t, 1 + 2 * NbBlocks>;
		
		ArrayTy totalSizeNbEltsAndOffsets;
		
		std::copy(std::begin(nbEltsInEachBlock), std::end(nbEltsInEachBlock), std::begin(totalSizeNbEltsAndOffsets) + 1);
		
		std::size_t totalSize = (std::tuple_size_v<ArrayTy> * sizeof(std::size_t) + BlockAlignment - 1) & ~(BlockAlignment - 1);
		
		SpUtils::foreach_index(
		[](auto&& index) {
			totalSizeNbEltsAndOffsets[1 + NbBlocks + index] = totalSize;
			
			if constexpr(index < NbBlocks) {
				using BlockTy = std::tuple_element_t<index, TupleTy>;
				totalSize += BlockTy::getSize(std::get<index>(nbEltsInEachBlock));
			}
			
		}, std::make_index_sequence<NbBlocks+1>{});
		
		// TO DO : for posix mem align totalSize should be a multiple of sizeof(void *)
		// static_assert(sizeof(void*) is a power of 2); 
		// if (posix)
		//     totalSize  = (totalSize + sizeof(void*) - 1) & ~(sizeof(void*) - 1);  
		
		totalSizeNbEltsAndOffsets[0] = totalSize;
		
		this->allocateBuffer(totalSize);
		
		std::memcpy(this->buffer, offsetsAndTotalSize.data(), (1 + 2 * NbBlocks) * sizeof(std::size_t));
	}
	
	std::size_t getTotalAllocatedSize() const {
		return *(this->getTotalSizeNbEltsOffsetsBegin());
	}
	
	template <std::size_t index>
	std::size_t getNbEltsInBlock() const {
		static_assert(index >= 0 && index < NbBlocks);
		this->getNbEltsBegin()[index];
	}
	
	template <std::size_t index>
	auto getBlockBegin() const {
		static_assert(index >= 0 && index < NbBlocks);
		static_cast<void*>(static_cast<char*>(this->buffer) + this->getOffsetsBegin()[index]);
	}
	
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
};

#endif
