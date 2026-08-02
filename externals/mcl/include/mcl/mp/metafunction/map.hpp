// This file is part of the mcl project.
// Copyright (c) 2022 merryhime
// SPDX-License-Identifier: MIT

#pragma once

#include "mcl/mp/typelist/list.hpp"

namespace mcl::mp {

namespace detail {

template<template<class...> class F, class L>
struct map_impl;

template<template<class...> class F, template<class...> class LT, class... Es>
struct map_impl<F, LT<Es...>> {
    using type = LT<F<Es>...>;
};

#if defined(__clang__) && !defined(_MSC_VER)
template<template<class...> class F, class... Es>
struct map_impl<F, mcl::mp::list<Es...>> {
    using type = mcl::mp::list<F<Es>...>;
};
#endif

}  // namespace detail

/// Applies each element of list L to metafunction F
template<template<class...> class F, class L>
using map = typename detail::map_impl<F, L>::type;

}  // namespace mcl::mp
