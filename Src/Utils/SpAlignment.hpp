#ifndef SPALIGNMENT_HPP
#define SPALIGNMENT_HPP

#include <type_traits>

template <const std::size_t alignment>
struct SpAlignment {
	static_assert((alignment != 0) && ((alignment & (alignment - 1)) == 0));
	
	static constexpr auto value = alignment;
};

template <class T>
struct is_instantiation_of_sp_alignment : std::false_type {};

template <const std::size_t alignment>
struct is_instantiation_of_sp_alignment<SpAlignment<alignment>> : std::true_type {};

#endif
