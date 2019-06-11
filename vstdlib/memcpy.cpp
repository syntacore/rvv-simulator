/**
    @file memcpy.cpp
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
memcpy(void * const dest, void const *src, size_t count)
{
    if (0 != count) {
        int8_t const *ps = static_cast<int8_t const*>(src);
        int8_t *pd = static_cast<int8_t *>(dest);
        do {
            size_t const vl = vsetvli(count, vtypei(e8, m8));
            count -= vl;
            vlb_v(v0, ps);
            ps += vl;
            vsb_v(v0, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}
}  // namespace rvv
