/**
    @file memmove.cpp
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
memmove(void *const dest, void const *src, size_t count)
{
    uint8_t const *ps = static_cast<uint8_t const*>(src);
    uint8_t *pd = static_cast<uint8_t *>(dest);
    if (0 != count || pd == ps) {
        if (pd < ps || ps + count <= pd) {
            return memcpy(dest, src, count);
        } else {
            memcpy_backward(pd, ps, count);
            return dest;
        }
    }

    return dest;
}
}  // namespace rvv
