/**
    @file vector_examples.cpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief RISCV-V Vector extension (v0.7) simulator usage examples and tests
*/

#define BOOST_TEST_MODULE vector_examples

#include <boost/test/included/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "riscv/ext/v.hpp"

using namespace ::riscv::v;

namespace {
std::default_random_engine generator(0);

void*
vvaddint32(size_t n, void * const dest, void const *src_a, void const *src_b)
{
    if (0 != n) {
        int32_t const *pa = static_cast<int32_t const*>(src_a);
        int32_t const *pb = static_cast<int32_t const*>(src_b);
        int32_t *pd = static_cast<int32_t*>(dest);

        do {
            size_t const vl = vsetvli(n, vtypei(e32, m1)); // Set vector length based on 32-bit vectors
            vlw_v(v0, pa);        // Get first vector
            n -= vl;              // Decrement number done
            pa += vl;             // Bump pointer
            vlw_v(v1, pb);        // Get second vector
            pb += vl;             // Bump pointer
            vadd_vv(v2, v0, v1);  // Sum vectors
            vsw_v(v2, pd);        // Store result
            pd += vl;             // Bump pointer
        } while (n);              // Loop back
    }

    return dest;
}

void
mixed_width(size_t n, void const *a, void * const b, void const *c)
{
    if (0 != n) {
        int8_t const *pa = static_cast<int8_t const*>(a);
        int32_t *pb = static_cast<int32_t*>(b);
        int32_t const *pc = static_cast<int32_t const*>(c);

        do {
            size_t vl = vsetvli(n, vtypei(e8, m1)); // Byte vector for predicate calc
            vlb_v(v1, pa);                      // Load a[i]
            pa += vl;                           // Bump pointer
            vmsle_vi(v0, v1, 5 - 1);            // a[i] < 5?

            vl = vsetvli(n, vtypei(e32, m4));   // Vector of 32-bit values
            n -= vl;                            // Decrement count
            vmv_v_i(v4, 1);                     // Splat immediate to destination
            vlw_v(v4, pc, vop_type::masked_in); // Load requested elements of C
            pc += vl;                           // Bump pointer
            vsw_v(v4, pb);                      // Store b[i]
            pb += vl;                           // Bump pointer
        } while (n);                            // Any more?
    }
}

void*
vmemcpy(void * const dest, void const *src, size_t n)
{
    if (0 != n) {
        int8_t const *ps = static_cast<int8_t const*>(src);
        int8_t *pd = static_cast<int8_t *>(dest);
        do {
            size_t const vl = vsetvli(n, vtypei(e8, m8)); // Vectors of 8b
            vlb_v(v0, ps);     // Load bytes
            ps += vl;          // Bump pointer
            n -= vl;           // Decrement count
            vsb_v(v0, pd);     // Store bytes
            pd += vl;          // Bump pointer
        } while (n);           // Any more?
    }

    return dest;
}

void
conditional(size_t n, void const *x, void const *a, void const *b, void * const z)
{
    if (0 != n) {
        int8_t const *px = static_cast<int8_t const*>(x);
        int16_t const *pa = static_cast<int16_t const*>(a);
        int16_t const *pb = static_cast<int16_t const*>(b);
        int16_t *pz = static_cast<int16_t*>(z);

        do {
            size_t const vl = vsetvli(n, vtypei(e16)); // Use 16b elements.
            vlb_v(v0, px);           // Get x[i], sign-extended to 16b
            n -= vl;                 // Decrement element count
            px += vl;                // x[i] Bump pointer
            vmsle_vi(v0, v0, 5 - 1); // Set mask in v0
            vlh_v(v1, pa, vop_type::masked_in); // z[i] = a[i] case
            vmnot_m(v0, v0);         // Invert v0
            pa += vl;                // a[i] bump pointer
            vlh_v(v1, pb, vop_type::masked_in); // z[i] = b[i] case
            pb += vl;                // b[i] bump pointer
            vsh_v(v1, pz);           // Store z
            pz += vl;                // b[i] bump pointer
        } while (n);
    }
}

void
saxpy(size_t n, float const a, float const *x, float *y)
{
    if (0 != n) {
        do {
            size_t const vl = vsetvli(n, vtypei(e32, m8));
            vlw_v(v0, reinterpret_cast<int32_t const*>(x));
            n -= vl;
            x += vl;
            vlw_v(v8, reinterpret_cast<int32_t*>(y));
            vfmacc_vf(v8, a, v0);
            vsw_v(v8, reinterpret_cast<int32_t*>(y));
            y += vl;
        } while (n);
    }
}

void
sgemm(size_t n, size_t m, size_t k, float const *a, size_t lda, float const *b, size_t ldb, float *c, size_t ldc)
{
    if ((n == 0) || (m == 0) || (k == 0)) {
        return;
    }

    size_t const astride = lda;
    size_t const bstride = ldb;
    size_t const cstride = ldc;

    while (m >= 16) { // Loop across rows of C blocks
        size_t nt = n; // Initialize n counter for next row of C blocks
        float const *bnp = b; // Initialize B n-loop pointer to start
        float *cnp = c; // Initialize C n-loop pointer

        while (nt) { // Loop across one row of C blocks
            size_t const nvl = vsetvli(nt, vtypei(e32)); // 32-bit vectors, LMUL=1
            float const *akp = a; // reset pointer into A to beginning
            float const *bkp = bnp; // step to next column in B matrix

            // Initialize current C submatrix block from memory
            vlw_v(v0, reinterpret_cast<int32_t*>(cnp));
            float *ccp = cnp + cstride;
            vlw_v(v1, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v2, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v3, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v4, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v5, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v6, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v7, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v8, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v9, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v10, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v11, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v12, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v13, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v14, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vlw_v(v15, reinterpret_cast<int32_t*>(ccp));

            size_t kt = k; // Initialize inner loop counter

            // Inner loop scheduled assuming 4-clock occupancy of vfmacc instruction and single-issue pipeline
            // Software pipeline loads
            float ft0 = akp[0];
            float const *amp = akp + astride;
            float ft1 = amp[0];
            amp += astride;
            float ft2 = amp[0];
            amp += astride;
            float ft3 = amp[0];
            amp += astride;

            float ft15 = 0;

            // Get vector from B matrix
            vlw_v(v16, reinterpret_cast<int32_t const*>(bkp));

            while (kt) { // Loop on inner dimension for current C block
                vfmacc_vf(v0, ft0, v16);
                bkp += bstride;
                float ft4 = amp[0];
                amp += astride;
                vfmacc_vf(v1, ft1, v16);
                kt -= 1; // Decrement k counter
                float ft5 = amp[0];
                amp += astride;
                vfmacc_vf(v2, ft2, v16);
                float ft6 = amp[0];
                amp += astride;
                float ft7 = amp[0];
                vfmacc_vf(v3, ft3, v16);
                amp += astride;
                float ft8 = amp[0];
                amp += astride;
                vfmacc_vf(v4, ft4, v16);
                float ft9 = amp[0];
                amp += astride;
                vfmacc_vf(v5, ft5, v16);
                float ft10 = amp[0];
                amp += astride;
                vfmacc_vf(v6, ft6, v16);
                float ft11 = amp[0];
                amp += astride;
                vfmacc_vf(v7, ft7, v16);
                float ft12 = amp[0];
                amp += astride;
                vfmacc_vf(v8, ft8, v16);
                float ft13 = amp[0];
                amp += astride;
                vfmacc_vf(v9, ft9, v16);
                float ft14 = amp[0];
                amp += astride;
                vfmacc_vf(v10, ft10, v16);
                float ft15 = amp[0];
                amp += astride;
                akp += 1; // Move to next column of a
                vfmacc_vf(v11, ft11, v16);
                // Don't load past end of matrix
                if (0 != kt) {
                    ft0 = akp[0];
                    amp = akp + astride;
                }
                vfmacc_vf(v12, ft12, v16);
                if (0 != kt) {
                    ft1 = amp[0];
                    amp += astride;
                }
                vfmacc_vf(v13, ft13, v16);
                if (0 != kt) {
                    ft2 = amp[0];
                    amp += astride;
                }
                vfmacc_vf(v14, ft14, v16);
                if (0 != kt) {
                    ft3 = amp[0];
                    amp += astride;
                }
                vfmacc_vf(v15, ft15, v16);
                vlw_v(v16, reinterpret_cast<int32_t const *>(bkp));
            } // k_loop
            vfmacc_vf(v15, ft15, v16);

            // Save C matrix block back to memory
            vsw_v(v0, reinterpret_cast<int32_t*>(cnp));
            ccp = cnp + cstride;
            vsw_v(v1, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v2, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v3, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v4, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v5, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v6, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v7, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v8, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v9, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v10, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v11, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v12, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v13, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v14, reinterpret_cast<int32_t*>(ccp));
            ccp += cstride;
            vsw_v(v15, reinterpret_cast<int32_t*>(ccp));

            // Following tail instructions should be scheduled earlier in free slots during C block save

            //  Bump pointers for loop across blocks in one row
            cnp += nvl; // Move C block pointer over
            bnp += nvl; // Move B block pointer over
            nt -= nvl; // Decrement element count in n dimension
        } // c_col_loop

        m -= 16;
        a += astride * 16;
        c += cstride * 16;
    } // c_row_loop
    // TODO: Handle end of matrix with fewer than 16 rows.
}
}  // namespace

BOOST_AUTO_TEST_CASE(vector_vector_add_example)
{
    using std::begin;
    using std::end;
    typedef std::vector<int32_t> buf_type;
    const size_t buf_size = 128;
    buf_type in_buf_a;
    buf_type in_buf_b;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf_a), buf_size, gen);
    std::generate_n(std::back_inserter(in_buf_b), buf_size, gen);

    buf_type out_buf(in_buf_a.size());
    buf_type ref_buf;

    vvaddint32(buf_size, &out_buf[0], &in_buf_a[0], &in_buf_b[0]);
    std::transform(in_buf_a.begin(), in_buf_a.end(), in_buf_b.begin(), std::back_inserter(ref_buf), std::plus<int32_t>());

    BOOST_TEST(ref_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(mixed_width_example)
{
    using std::begin;
    using std::end;
    typedef std::vector<int32_t> buf_type;
    const size_t buf_size = 128;
    std::vector<int8_t> a;
    buf_type b;
    buf_type c;

    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static std::uniform_int_distribution<int16_t> distribution_a(0, 10);
    static auto const gen = []() { return distribution(generator); };
    static auto const gen_a = []() { return distribution_a(generator); };
    std::generate_n(std::back_inserter(a), buf_size, gen_a);
    std::generate_n(std::back_inserter(b), buf_size, gen);
    std::generate_n(std::back_inserter(c), buf_size, gen);

    buf_type out_buf(buf_size);
    buf_type ref_buf;

    for (size_t i = 0; i < buf_size; ++i) {
        ref_buf.push_back(a[i] < 5 ? c[i] : 1);
    }

    mixed_width(buf_size, &a[0], &b[0], &c[0]);

    BOOST_TEST(ref_buf == b);
}

BOOST_AUTO_TEST_CASE(memcpy_example)
{
    using std::begin;
    using std::end;
    typedef std::vector<char> buf_type;
    const size_t buf_size = 1024;
    buf_type in_buf;
    static std::uniform_int_distribution<int> distribution(0, 255);
    static auto const gen = []() {return distribution(generator); };
    std::generate_n(std::back_inserter(in_buf), buf_size, gen);

    buf_type out_buf(in_buf.size());
    vmemcpy(&out_buf[0], &in_buf[0], in_buf.size() * sizeof(buf_type::value_type));

    BOOST_TEST(in_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(conditional_example)
{
    using std::begin;
    using std::end;
    typedef std::vector<int16_t> buf_type;
    const size_t buf_size = 128;
    buf_type a;
    buf_type b;
    std::vector<int8_t> x;
    static std::uniform_int_distribution<buf_type::value_type> distribution(0, 255);
    static std::uniform_int_distribution<int16_t> distribution_x(0, 10);
    static auto const gen = []() { return distribution(generator); };
    static auto const gen_x = []() { return distribution_x(generator); };
    std::generate_n(std::back_inserter(x), buf_size, gen_x);
    std::generate_n(std::back_inserter(a), buf_size, gen);
    std::generate_n(std::back_inserter(b), buf_size, gen);

    buf_type out_buf(a.size());
    buf_type ref_buf;

    conditional(buf_size, &x[0], &a[0], &b[0], &out_buf[0]);

    for (size_t i = 0; i < buf_size; ++i) {
        ref_buf.push_back(x[i] < 5 ? a[i] : b[i]);
    }

    BOOST_TEST(ref_buf == out_buf);
}

BOOST_AUTO_TEST_CASE(saxpy_example)
{
    using std::begin;
    using std::end;
    typedef std::vector<float> buf_type;
    const size_t buf_size = 128;
    float const a = 2.7f;

    buf_type x;
    buf_type y;

    static std::uniform_real_distribution<float> distribution(0, 255);
    static auto const gen = []() { return distribution(generator); };
    std::generate_n(std::back_inserter(x), buf_size, gen);
    std::generate_n(std::back_inserter(y), buf_size, gen);

    buf_type ref_buf;
    for (size_t i = 0; i < buf_size; ++i) {
        ref_buf.push_back(a * x[i] + y[i]);
    }

    saxpy(buf_size, a, &x[0], &y[0]);

    BOOST_TEST(ref_buf == y);
}

BOOST_AUTO_TEST_CASE(sgemm_example)
{
    using std::begin;
    using std::end;

    const size_t M = 16;
    const size_t K = 16;
    const size_t N = 16;

    typedef std::vector<float> buf_type;

    buf_type a;
    buf_type b;
    buf_type c;

    static std::uniform_real_distribution<float> distribution(0, 10);
    static auto const gen = []() { return distribution(generator); };

    std::generate_n(std::back_inserter(a), M * K, gen);
    std::generate_n(std::back_inserter(b), K * N, gen);
    std::generate_n(std::back_inserter(c), M * N, gen);

    buf_type ref_buf(M * N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = c[i * N + j];
            for (size_t k = 0; k < K; k++)
                sum += a[i * K + k] * b[k * N + j];
            ref_buf[i * N + j] = sum;
        }
    }

    sgemm(N, M, K, &a[0], M, &b[0], K, &c[0], M);

    BOOST_TEST(ref_buf == c);
}
