#ifndef SPALIGNMENT_HPP
#define SPALIGNMENT_HPP

struct SpAlignment {
public:
	
	constexpr explicit SpAlignment(std::size_t inAlignment) {
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

#endif
