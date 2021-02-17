#ifndef SPALIGNMENT_HPP
#define SPALIGNMENT_HPP

template <unsigned long long alignment>
struct SpAlignment {
	constexpr unsigned long long operator() const {
		return alignment;
	}
	
	constexpr bool isPowerOf2() const {
		return (alignment != 0ULL) && ((alignment & (alignment - 1)) == 0ULL);
	}
	
	constexpr SpAlignment() {
		static_assert(isPowerOf2());
	}
};

#endif
