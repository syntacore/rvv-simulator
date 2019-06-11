/**
    @file unit_tests.cpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief Vector extension simulator (v0.7) per-instruction tests
*/

#define BOOST_TEST_MODULE basic_ops

#include <boost/test/included/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "riscv/ext/v.hpp"

using namespace ::riscv::v;

namespace {
std::default_random_engine generator(0);

template<typename Ty>
void*
add(void * const dest, void const *src_a, void const *src_b, size_t count)
{
    if (0 != count) {
        Ty const *pa = static_cast<Ty const*>(src_a);
        Ty const *pb = static_cast<Ty const*>(src_b);
        Ty *pd = static_cast<Ty*>(dest);

        do {
            size_t const vl = vsetvl(count, vtype(e32, m8));
            count -= vl;
            vlw_v(v0, pa);
            pa += vl;
            vlw_v(v8, pb);
            pb += vl;
            vadd_vv(v16, v8, v0);
            vsw_v(v16, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}

template<typename Ty>
void*
addx(void * const dest, void const *src_a, int32_t const src_b, size_t count)
{
    if (0 != count) {
        Ty const *pa = static_cast<Ty const*>(src_a);
        Ty *pd = static_cast<Ty*>(dest);

        do {
            size_t const vl = vsetvl(count, vtype(e32, m8));
            count -= vl;
            vlw_v(v0, pa);
            pa += vl;
            vadd_vx(v16, v0, src_b);
            vsw_v(v16, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}

template<typename Ty>
void*
addi(void * const dest, void const *src_a, int16_t const imm, size_t count)
{
    if (0 != count) {
        Ty const *pa = static_cast<Ty const*>(src_a);
        Ty *pd = static_cast<Ty*>(dest);

        do {
            size_t const vl = vsetvl(count, vtype(e32, m8));
            count -= vl;
            vlw_v(v0, pa);
            pa += vl;
            vadd_vi(v16, v0, imm);
            vsw_v(v16, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}

template<typename Ty>
void*
sub(void * const dest, void const *src_a, void const *src_b, size_t count)
{
    if (0 != count) {
        Ty const *pa = static_cast<Ty const*>(src_a);
        Ty const *pb = static_cast<Ty const*>(src_b);
        Ty *pd = static_cast<Ty*>(dest);

        do {
            size_t const vl = vsetvl(count, vtype(e32, m8));
            count -= vl;
            vlw_v(v0, pa);
            pa += vl;
            vlw_v(v8, pb);
            pb += vl;
            vsub_vv(v16, v8, v0);
            vsw_v(v16, pd);
            pd += vl;
        } while (count);
    }

    return dest;
}
}  // namespace

BOOST_AUTO_TEST_CASE(addition)
{
    using std::begin;
    using std::end;
    typedef std::vector<int32_t> buf_type;
    const size_t buf_size = 16;
    buf_type in_buf_a;
    buf_type in_buf_b;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf_a), buf_size, gen);
    std::generate_n(std::back_inserter(in_buf_b), buf_size, gen);

    buf_type out_buf(in_buf_a.size());
    buf_type ref_buf;

    add<buf_type::value_type>(&out_buf[0], &in_buf_a[0], &in_buf_b[0], in_buf_a.size());
    std::transform(in_buf_a.begin(), in_buf_a.end(), in_buf_b.begin(), std::back_inserter(ref_buf), std::plus<int32_t>());

    BOOST_TEST(ref_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(addition_scalar)
{
    using std::begin;
    using std::end;
    namespace ph = std::placeholders;
    typedef std::vector<int32_t> buf_type;
    size_t const buf_size = 16;
    int32_t const x = 127;
    buf_type in_buf_a;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf_a), buf_size, gen);

    buf_type out_buf(in_buf_a.size());
    buf_type ref_buf;

    addx<buf_type::value_type>(&out_buf[0], &in_buf_a[0], x, in_buf_a.size());
    std::transform(in_buf_a.begin(), in_buf_a.end(), std::back_inserter(ref_buf), std::bind(std::plus<int32_t>(), ph::_1, x));

    BOOST_TEST(ref_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(addition_immediate)
{
    using std::begin;
    using std::end;
    namespace ph = std::placeholders;
    typedef std::vector<int32_t> buf_type;
    size_t const buf_size = 16;
    int16_t const imm = 127;
    buf_type in_buf_a;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf_a), buf_size, gen);

    buf_type out_buf(in_buf_a.size());
    buf_type ref_buf;

    addi<buf_type::value_type>(&out_buf[0], &in_buf_a[0], imm, in_buf_a.size());
    std::transform(in_buf_a.begin(), in_buf_a.end(), std::back_inserter(ref_buf), std::bind(std::plus<int32_t>(), ph::_1, imm));

    BOOST_TEST(ref_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(subtraction)
{
    using std::begin;
    using std::end;
    typedef std::vector<int32_t> buf_type;
    const size_t buf_size = 32;
    buf_type in_buf_a;
    buf_type in_buf_b;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf_a), buf_size, gen);
    std::generate_n(std::back_inserter(in_buf_b), buf_size, gen);

    buf_type out_buf(in_buf_a.size());
    buf_type ref_buf;

    sub<buf_type::value_type>(&out_buf[0], &in_buf_a[0], &in_buf_b[0], in_buf_a.size());
    std::transform(in_buf_a.begin(), in_buf_a.end(), in_buf_b.begin(), std::back_inserter(ref_buf), std::minus<int32_t>());

    BOOST_TEST(ref_buf == out_buf);
}
