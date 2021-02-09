#ifndef SPBLOCKVECTOR_HPP
#define SPBLOCKVECTOR_HPP

#include <cstdef>
#include <cstdlib>

struct SpBlockAlignment {
public:
	template <class... Blocks>
	constexpr explicit SpBlockAlignment() : alignment(alignof(Blocks::value_type...)) {} 
	
	constexpr explicit SpBlockAlignment(std::size_t inAlignment) {
		alignment = inAlignment;
	}
	
	constexpr std::size_t operator() const {
		return alignment;
	}
	
	constexpr bool isPowerOf2() const {
		return (alignment != 0) && ((alignment & (alignment - 1)) == std::size_t(0));
	}
	
private:
	std::size_t alignment;
};

template <class... Blocks, const SpBlockAlignment alignment=SpBlockTupleAlignment<std::size_t, Blocks...>{}>
class SpBlockVector {
	static_assert(sizeof...(Blocks) >= 1);
	static_assert(alignment.isPowerOf2());
	// posix memalign check : if over alignment then allocated size should be a multiple of sizeof(void*)
	static_assert(std::conjunction_v<std::bool_constant<alignof(std::size_t) <= alignment>, std::bool_constant<alignof(Blocks::value_type) <= alignment>...>);
	static_assert(std::conjunction_v<std::is_trivially_copyable<Blocks::value_type>...>);

private:
	void allocateBuffer(std::size_t size) {
		if constexpr (alignof(std::max_align_t) >= alignment) {
			buffer = std::malloc(size);
		} else {
			buffer = std::aligned_alloc(size, alignment);
		}
	}
	
	void deallocateBuffer() {
		if(buffer) {
			std::free(buffer);
		}
	}

public:

	explicit SpBlockVector(std::array<std::size_t, sizeof...(Blocks)> nbEltsInEachBlock) {
		
	}
	
private:
	std::size_t capacity;
	void* buffer;
};

#endif
