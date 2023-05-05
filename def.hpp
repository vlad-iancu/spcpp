#ifndef SPCPP_DEF_H
#define SPCPP_DEF_H

#include <cstdint>

namespace spcpp
{
	using real = double;
	using size = std::size_t;
	using i64 = std::int64_t;
	using u64 = std::uint64_t;
	using i32 = std::int32_t;
	using u32 = std::uint32_t;
	using i16 = std::int16_t;
	using u16 = std::uint16_t;

	enum order
	{
		SPCPP_COL_MAJOR,
		SPCPP_ROW_MAJOR
	};
}

#endif
