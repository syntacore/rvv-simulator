/**
    @file memcpy_backward.cpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief Vector extension simulator (v0.7) example
*/

#include "vstdlib/vstdlib.hpp"

#include <cassert>

namespace rvv {

using namespace ::riscv::v;

void*
memcpy_backward(void *const dest, void const *src, size_t count)
{
    if (0 != count) {
        int8_t const *ps = static_cast<int8_t const*>(src) + (count - 1);
        int8_t *pd = static_cast<int8_t *>(dest) + (count - 1);

        static ptrdiff_t const stride = -1;
        do {
            size_t const vl = vsetvli(count, vtypei(e8, m8));
            count -= vl;
            vlsb_v(v0, ps, stride);
            ps -= vl;
            vssb_v(v0, pd, stride);
            pd -= vl;
        } while (count);
    }
    return dest;
}
}  // namespace rvv
