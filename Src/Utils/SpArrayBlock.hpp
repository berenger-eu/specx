#ifndef SPARRAYBLOCK_HPP
#define SPARRAYBLOCK_HPP

template <typename DataType, const SpAlignment alignment=SpAlignment<alignof(DataType)>{}>
class SpArrayBlock {
	static_assert(std::is_trivially_copyable_v<DataType>);
	static_assert(static_cast<unsigned long long>(alignment) >= alignof(DataType));
	
private:
	DataType* data;
	unsigned long long inNbElts;
	 
public:
	using value_type = DataType;
	
	SpArrayBlock() : data(nullptr), nbElts(0ULL) {}
	SpArrayBlock(void* inData, unsigned long nbElts) :data(inData), nbElts(inNbElts) {}
	
	static constexpr auto getAlignment() {
		return static_cast<unsigned long long>(alignment);
	}
	
	static constexpr unsigned long long getSize(unsigned long long nbElts, unsigned long long alignment) {
		return (nbElts * sizeof(Datatype) + alignment - 1ULL) & ~(alignment - 1ULL);
	}
	
	auto begin() {
		return begin;
	}
	
	auto end() {
		return begin + nbElts;
	}
};

#endif
