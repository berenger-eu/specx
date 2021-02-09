#ifndef SPVECTORBLOCK_HPP
#define SPVECTORBLOCK_HPP

template <class DataType>
class SpVectorBlock {
public:
	using value_type = DataType;
	SpVectorBlock() = default;
	
	constexpr static std::size_t getSize(std::size_t nbElts) {
		return nbElts * sizeof(DataType);
	}
	
	class Viewer {
		private:
			const std::size_t nbElts;
			DataType* ptr;
		
		public:
			std::size_t size() const {
				return nbElts;
			}
			
			DataType& operator[](std::size_t index) {
				assert(index < nbElts);
				return ptr[index];
			}
		
	};
};

#endif
