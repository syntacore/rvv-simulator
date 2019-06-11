/**
    @file vstdlib.hpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief Vector extension simulator (v0.7) example
*/

#ifndef RISCV_VEXAMPLES_HPP_
#define RISCV_VEXAMPLES_HPP_

#include "riscv/ext/v.hpp"

#include <type_traits>
#include <iterator>

namespace rvv {

void *memset(void *const dest, int c, size_t count);
void *memcpy(void * const dest, void const *src, size_t count);
void *memmove(void *const dest, void const *src, size_t count);
void *memcpy_backward(void *pd, void const *ps, size_t count);

}  // namespace rvv

#endif  // RISCV_VEXAMPLES_HPP_
