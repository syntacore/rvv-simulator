/**
    @file memset.cpp
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
memset(void *const dest, int c, size_t count)
{
    if (0 != count) {
        vsetvli(count, vtypei(e8, m8));
        vmv_v_x(v0, c);

        int8_t *pd = static_cast<int8_t *>(dest);
        do {
            size_t const vl = vsetvli(count, vtypei(e8, m8));
            count -= vl;
            vsb_v(v0, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}

}  // namespace rvv
