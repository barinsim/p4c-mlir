/*
Copyright 2013-present Barefoot Networks, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/* -*-C++-*- */

#ifndef _LIB_NULL_H_
#define _LIB_NULL_H_

#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "error.h"  // for BUG macro

// Typical C contortions to transform something into a string
#define LIB_STRINGIFY(x) #x
#define LIB_TOSTRING(x) LIB_STRINGIFY(x)

#define CHECK_1_NULL_IMPL(r, unused, idx, a)  \
        if ((a) == nullptr) BUG(__FILE__ ":" LIB_TOSTRING(__LINE__) ": Null " #a);

#define CHECK_NULL(...)                                                                       \
    do {                                                                                      \
        BOOST_PP_SEQ_FOR_EACH_I(CHECK_1_NULL_IMPL, %%, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__)) \
    } while (0)

#endif /* _LIB_NULL_H_ */

