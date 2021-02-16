#ifndef SPARRAYBLOCK_HPP
#define SPARRAYBLOCK_HPP

template <typename DataType, const SpAlignment alignment=SpAlignment{alignof(DataType)}>
class alignas(static_cast<std::size_t>(alignment)) SpArrayBlock {
	static_assert(std::is_trivially_copyable_v<DataType>);
	static_assert(alignment.isPowerOf2());
	static_assert(static_cast<std::size_t>(alignment) >= alignof(DataType));
	
private:
	DataType* data;
	std::size_t inNbElts;
	 
public:
	using value_type = DataType;
	
	SpArrayBlock() : data(nullptr), nbElts(0) {}
	SpArrayBlock(void* inData, std:size_t nbElts) :data(inData), nbElts(inNbElts) {}
	
	static constexpr std::size_t getAlignment() {
		return static_cast<std::size_t>(alignment);
	}
	
	static constexpr std::size_t getSize(std::size_t nbElts, std::size_t alignment) {
		return (nbElts * sizeof(Datatype) + alignment - 1) & ~(alignment - 1);
	}
	
	auto begin() {
		return begin;
	}
	
	auto end() {
		return begin + nbElts;
	}
};

#endif
