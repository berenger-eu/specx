#ifndef SPSIMPLETUPLE_HPP
#define SPSIMPLETUPLE_HPP
template <typename First, typename... Rest>
struct SpSimpleTuple : public SpInternalBlockTuple<Rest...> {
	First first;
};

template<>
struct SpSimpleTuple<> {};

template <unsigned long long index, typename First, typename... Rest>
struct SpSimpleTupleGetImpl {
	static decltype(auto) value(SpInternalBlockTuple<First, Rest...>* t) {
		return SpSimpleTupleGetImpl<index-1, Rest...>(t);
	}
};

template <typename First, typename... Rest>
struct SpSimpleTupleGetImpl {
	static decltype(auto) value(SpInternalBlockTuple<First, Rest...>* t) {
		return SpSimpleTupleGetImpl<index-1, Rest...>(t);
	}
};

template <unsigned long long index, typename First, typename... Rest>
inline decltype(auto) SpSimpleTupleGet(SpInternalBlockTuple<First, Rest...>& t) {
	static_assert(index >= 0 && index < unsigned long long(sizeof...(Rest) + 1));
	return SpSimpleTupleGetImpl<index, First, Rest...>::value(&t);
}
#endif
