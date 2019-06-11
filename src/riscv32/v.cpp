/**
    @file v.cpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief Vector extension simulator (v0.7)
*/

#include "riscv/ext/v.hpp"

#include <memory>
#include <algorithm>
#include <cassert>
#include <array>
#include <vector>
#include <type_traits>
#include <functional>
#include <cmath>
#include <limits>
#include <climits>

#ifdef _MSC_VER
#pragma warning( disable : 4250)
#endif

namespace riscv {
namespace v {
namespace spec_0_7 {
namespace implementation {
namespace {

template<size_t N>
struct Size_traits;

template<>
struct Size_traits<1u>
{
    typedef int8_t int_type;
    typedef uint8_t uint_type;
};

template<>
struct Size_traits<2u>
{
    typedef int16_t int_type;
    typedef uint16_t uint_type;
};

template<>
struct Size_traits<4u>
{
    typedef int32_t int_type;
    typedef uint32_t uint_type;
    typedef float32_t float_type;
};

template<>
struct Size_traits<8u>
{
    typedef int64_t int_type;
    typedef uint64_t uint_type;
    typedef float64_t float_type;
};

template<typename Ty>
inline
typename std::enable_if<std::is_floating_point<Ty>::value, Ty>::type
fsgnj(Ty const &s1, Ty const &s2)
{
    typedef Size_traits<sizeof(Ty)> size_traits;
    typedef typename size_traits::uint_type uint_type;
    typedef typename size_traits::int_type int_type;
    static uint_type const mask1 = uint_type((std::numeric_limits<int_type>::max)());
    auto const p1 = reinterpret_cast<uint_type const &>(s1);
    auto const p2 = reinterpret_cast<uint_type const &>(s2);
    auto const res = (mask1 & p1) | (~mask1 & p2);
    auto const fres = reinterpret_cast<Ty const &>(res);
    return fres;
}

template<typename Ty>
inline
typename std::enable_if<std::is_floating_point<Ty>::value, Ty>::type
fsgnjn(Ty const &s1, Ty const &s2)
{
    typedef Size_traits<sizeof(Ty)> size_traits;
    typedef typename size_traits::uint_type uint_type;
    typedef typename size_traits::int_type int_type;
    static uint_type const mask1 = uint_type((std::numeric_limits<int_type>::max)());
    auto const p1 = reinterpret_cast<uint_type const &>(s1);
    auto const p2 = reinterpret_cast<uint_type const &>(s2);
    auto const res = (mask1 & p1) | (~mask1 & ~p2);
    auto const fres = reinterpret_cast<Ty const &>(res);
    return fres;
}

template<typename Ty>
inline
typename std::enable_if<std::is_floating_point<Ty>::value, Ty>::type
fsgnjx(Ty const &s1, Ty const &s2)
{
    typedef Size_traits<sizeof(Ty)> size_traits;
    typedef typename size_traits::uint_type uint_type;
    typedef typename size_traits::int_type int_type;
    static uint_type const mask1 = uint_type((std::numeric_limits<int_type>::max)());
    auto const p1 = reinterpret_cast<uint_type const &>(s1);
    auto const p2 = reinterpret_cast<uint_type const &>(s2);
    auto const res = (mask1 & p1) ^(~mask1 & p2);
    auto const fres = reinterpret_cast<Ty const &>(res);
    return fres;
}

template<typename Ty>
inline
typename std::enable_if<
    std::is_integral<Ty>::value &&
    std::is_signed<Ty>::value &&
    std::is_integral<typename Size_traits<2 * sizeof(Ty)>::int_type>::value,
    Ty>::type
mulh(Ty const &x, Ty const &y)
{
    typedef typename Size_traits<2 * sizeof(Ty)>::int_type dbl_type;
    return static_cast<Ty>((dbl_type(x) * dbl_type(y)) >> (CHAR_BIT * sizeof(Ty)));
}

template<typename Ty1, typename Ty2>
inline
typename std::enable_if<
    sizeof(Ty1) == sizeof(Ty2) &&
    std::is_integral<Ty1>::value &&
    std::is_signed<Ty1>::value &&
    std::is_integral<Ty2>::value &&
    std::is_unsigned<Ty2>::value &&
    std::is_integral<typename Size_traits<2 * sizeof(Ty1)>::int_type>::value,
    Ty1>::type
mulhsu(Ty1 const &x, Ty2 const &y)
{
    typedef typename Size_traits<2 * sizeof(Ty1)>::int_type idbl_type;
    typedef typename Size_traits<2 * sizeof(Ty2)>::uint_type udbl_type;
    return static_cast<Ty1>((idbl_type(x) * idbl_type(udbl_type(y))) >> (CHAR_BIT * sizeof(Ty1)));
}

template<typename Ty>
inline
typename std::enable_if<
    std::is_integral<Ty>::value &&
    std::is_unsigned<Ty>::value &&
    std::is_integral<typename Size_traits<2 * sizeof(Ty)>::uint_type>::value,
    Ty>::type
mulhu(Ty const &x, Ty const &y)
{
    typedef typename Size_traits<2 * sizeof(Ty)>::uint_type udbl_type;
    return static_cast<Ty>((udbl_type(x) * udbl_type(y)) >> (CHAR_BIT * sizeof(Ty)));
}

inline int64_t
mulh(int64_t const &, int64_t const &)
{
    throw Instruction_undefined_for_element_size(sizeof(int64_t));
}

inline int64_t
mulhsu(int64_t const &, int64_t const &)
{
    throw Instruction_undefined_for_element_size(sizeof(int64_t));
}

inline int64_t
mulhu(int64_t const &, int64_t const &)
{
    throw Instruction_undefined_for_element_size(sizeof(int64_t));
}

class Impl_base
    : virtual public V_unit
{
public:
    virtual size_t setvl(size_t vl) = 0;
    virtual size_t setvstart(size_t vstart) = 0;
    virtual void setill(bool ill) = 0;
    virtual void setew(size_t ew) = 0;
    virtual void setmul(size_t mul) = 0;
#if 0
    virtual void set_mask_reg(vreg_no) = 0;
#endif
};
}  // namespace

class State
    : virtual public Impl_base
{
public:
    virtual size_t vl()const = 0;
    virtual size_t vstart()const = 0;
    virtual size_t vlmax()const = 0;
    virtual char* elt_ptr(vreg_no _reg, size_t _ind) = 0;
    virtual char const* elt_ptr(vreg_no _reg, size_t _ind)const = 0;
    virtual size_t sew()const = 0;
    virtual size_t lmul()const = 0;
    virtual bool is_ill()const = 0;
    virtual Operations& get_op_performer()const = 0;
    virtual Float_operations& get_fop_performer()const = 0;
    virtual bool is_enabled(size_t i)const = 0;
    virtual bool get_mask(vreg_no _reg, size_t _ind)const = 0;
    virtual void set_mask(vreg_no _reg, size_t _ind, bool value) = 0;
protected:
    virtual bool is_valid_reg(vreg_no _reg)const = 0;
    virtual bool mask_bit(size_t i)const = 0;
};

namespace {
template<typename Element_type, typename Memory_type>
class Bad_load
    : public Loader<Memory_type>
{
    static_assert(sizeof(Element_type) < sizeof(Memory_type), "Bad type");

    void operator()(V_unit &st, vreg_no, Memory_type const *, ptrdiff_t, vop_type mode) final
    {
        throw Load_wider_value_to_narrowed_element(sizeof(Memory_type), sizeof(Element_type));
    }

    void operator()(V_unit &st, vreg_no, Memory_type const *, vreg_no, vop_type mode) final
    {
        throw Load_wider_value_to_narrowed_element(sizeof(Memory_type), sizeof(Element_type));
    }
};

template<typename Element_type, typename Memory_type>
class Good_load;

template<typename Element_type, typename Memory_type>
using Loader_impl =
typename std::conditional<
    (sizeof(Element_type) < sizeof(Memory_type)),
    Bad_load<Element_type, Memory_type>,
    Good_load<Element_type, Memory_type>
>::type;

template<typename Element_type>
class Operations_impl;

template<typename Element_type>
class Operations_essentials;

template<typename Element_type, typename Memory_type>
class Good_load
    : public Loader<Memory_type>
{
    static_assert(sizeof(Element_type) >= sizeof(Memory_type), "Bad type");

    using Loader<Memory_type>::to_element;

    void operator()(V_unit& vu, vreg_no vd, Memory_type const *rs1, ptrdiff_t rs2, vop_type mode) final
    {
        State& st = dynamic_cast<State&>(vu);
        auto p = reinterpret_cast<char const *>(rs1);
        auto const len = st.vl();

        for (size_t i = 0; i < len; ++i, p += rs2) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type *const addr = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
                *addr = static_cast<Element_type>(*reinterpret_cast<Memory_type const *>(p));
            }
        }
    }

    void operator()(V_unit& vu, vreg_no vd, Memory_type const *rs1, vreg_no idx, vop_type mode) final
    {
        State& st = dynamic_cast<State&>(vu);
        auto const p = reinterpret_cast<char const *>(rs1);
        auto const len = st.vl();

        for (size_t i = 0; i < len; ++i) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Memory_type *const addr = reinterpret_cast<Memory_type *>(st.elt_ptr(vd, i));
                size_t const stride = *reinterpret_cast<size_t *>(st.elt_ptr(idx, i));
                *addr = to_element(*reinterpret_cast<Memory_type const *>(p + stride));
            }
        }
    }
};

template<typename Element_type, typename Memory_type>
class Saver_impl
    : protected Saver<Memory_type>
{
    void operator()(V_unit& vu, vreg_no vs1, Memory_type *rs1, ptrdiff_t rs2, vop_type mode) const final
    {
        State& st = dynamic_cast<State&>(vu);
        auto p = reinterpret_cast<char *>(rs1);
        auto const len = st.vl();

        for (size_t i = 0; i < len; ++i, p += rs2) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Memory_type *const addr = reinterpret_cast<Memory_type *>(p);
                *addr = *reinterpret_cast<Memory_type *>(st.elt_ptr(vs1, i));
            }
        }
    }

    void operator()(V_unit& vu, vreg_no vs1, Memory_type *rs1, vreg_no idx, vop_type mode) const final
    {
        State& st = dynamic_cast<State&>(vu);
        auto const p = reinterpret_cast<char *>(rs1);
        auto const len = st.vl();

        for (size_t i = 0; i < len; ++i) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                size_t const stride = *reinterpret_cast<size_t *>(st.elt_ptr(idx, i));
                Memory_type *const addr = reinterpret_cast<Memory_type *>(p + stride);
                *addr = *reinterpret_cast<Memory_type *>(st.elt_ptr(vs1, i));
            }
        }
    }
};

template<typename Element_type, typename Memory_type>
class Get_loader
    : virtual Operations
    , Loader_impl<Element_type, Memory_type>
{
    operator Loader<Memory_type> &() final
    {
        return static_cast<Loader_impl<Element_type, Memory_type> &>(*this);
    }
};

template<typename Element_type, typename Memory_type>
class Get_saver
    : virtual Operations
    , Saver_impl<Element_type, Memory_type>
{
    operator Saver<Memory_type> &() final
    {
        return static_cast<Saver_impl<Element_type, Memory_type> &>(*this);
    }
};

template<typename Element_type>
class Get_mem_IO
    : virtual Operations

    , Get_loader<Element_type, int8_t>
    , Get_loader<Element_type, int16_t>
    , Get_loader<Element_type, int32_t>
    , Get_loader<Element_type, int64_t>

    , Get_loader<Element_type, uint8_t>
    , Get_loader<Element_type, uint16_t>
    , Get_loader<Element_type, uint32_t>

    , Get_saver<Element_type, int8_t>
    , Get_saver<Element_type, int16_t>
    , Get_saver<Element_type, int32_t>
    , Get_saver<Element_type, int64_t>
{
};

template<typename Element_type>
class Non_scalar_operations_essentials
{
protected:
    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type()>, Func>::value, void>::type
    iterate(V_unit& vu, Func &&func, vreg_no vd)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
            *dest_i = func();
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
                *dest_i = 0;
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(
        Element_type const &)>, Func>::value, void>::type
    iterate(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vop_type mode)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type const src1_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs1, i));
                *dest_i = func(src1_i);
            }
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
                *dest_i = 0;
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(Element_type const &,
                                                                          Element_type const &)>, Func>::value, void>::type
    iterate(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vreg_no vs2, vop_type mode)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type const src1_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs1, i));
                Element_type const src2_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs2, i));
                *dest_i = func(src1_i, src2_i);
            }
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
                *dest_i = 0;
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(Element_type const &, Element_type const &,
                                                                          Element_type const &)>, Func>::value, void>::type
    iterate(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3, vop_type mode)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type const src1_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs1, i));
                Element_type const src2_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs2, i));
                Element_type const src3_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs3, i));
                *dest_i = func(src1_i, src2_i, src3_i);
            }
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                Element_type *const dest_i = reinterpret_cast<Element_type *>(st.elt_ptr(vd, i));
                *dest_i = 0;
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(Element_type const &)>, Func>::value, void>::type
    iterate_vm(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vop_type mode)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type const src1_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs1, i));
                bool result = func(src1_i);
                st.set_mask(v0, i, result);
            }
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                st.set_mask(v0, i, false);
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(Element_type const &,
                                                                          Element_type const &)>, Func>::value, void>::type
    iterate_vm(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vreg_no vs2, vop_type mode)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            if (mode == vop_type::thread_all || mode == vop_type::masked_in && st.is_enabled(i)) {
                Element_type const src1_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs1, i));
                Element_type const src2_i = *reinterpret_cast<Element_type const *>(st.elt_ptr(vs2, i));
                bool result = func(src1_i, src2_i);
                st.set_mask(v0, i, result);
            }
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                st.set_mask(v0, i, false);
            }
        }
    }

    template<typename Func>
    typename std::enable_if<std::is_assignable<std::function<Element_type(Element_type const &,
                                                                          Element_type const &)>, Func>::value, void>::type
    iterate_mm(V_unit& vu, Func &&func, vreg_no vd, vreg_no vs1, vreg_no vs2)
    {
        State& st = dynamic_cast<State&>(vu);

        if (st.is_ill()) {
            throw State_not_configured();
        }

        auto const vstart = st.vstart();
        auto const vl = st.vl();

        for (size_t i = vstart; i < vl; ++i) {
            bool src1_i = st.get_mask(vs1, i);
            bool src2_i = st.get_mask(vs2, i);
            bool result = func(src1_i, src2_i);
            st.set_mask(v0, i, result);
        }

        if (vl) {
            for (size_t i = vl; i < st.vlmax(); ++i) {
                st.set_mask(v0, i, false);
            }
        }
    }
};


template<typename Element_type>
class Operations_essentials
    : virtual protected Operations
    , protected Non_scalar_operations_essentials<Element_type>
{
};

template<typename Element_type>
class Float_operations_essentials
    : virtual protected Float_operations
    , protected Non_scalar_operations_essentials<Element_type>
{
};

template<typename Element_type>
class Operations_impl
    : virtual public Operations
    , Get_mem_IO<Element_type>
    , Operations_essentials<Element_type>
{
private:
    virtual operator Operations &() final
    {
        return *this;
    }

#if 0
    template<typename RTy, typename Ty, typename Func>
    static
    typename std::enable_if<
        (sizeof(Element_type) < sizeof(Ty) || sizeof(Element_type) < sizeof(Ty))
        && std::is_assignable<std::function<RTy(Ty const &)>, Func>::value,
        std::function<Element_type(Element_type const &)>
    >::type
    adapter1(Func &&)
    {
        throw Load_wider_value_to_narrowed_element((std::max)(sizeof(Ty), sizeof(RTy)), sizeof(Element_type));
    }

    template<typename RTy, typename Ty, typename Func,
        typename = typename std::enable_if_t<
            (sizeof(Element_type) >= sizeof(Ty) && sizeof(Element_type) >= sizeof(RTy))
            && std::is_assignable<std::function<RTy(Ty const &)>, Func>::value>
    >
    static auto adapter1(Func &&func)
    {
        return
            [&func](Element_type const &x)->Element_type {
                return
                    to_element(func(
                        reinterpret_cast<Ty const &>(x)
                    ));
            };
    }

    template<typename RTy, typename Ty, typename Func>
    static
    typename std::enable_if<
        (sizeof(Element_type) < sizeof(Ty) || sizeof(Element_type) < sizeof(RTy))
        && std::is_assignable<std::function<RTy(Ty const &, Ty const &)>, Func>::value,
        std::function<Element_type(Element_type const &, Element_type const &)>
    >::type
    adapter2(Func &&)
    {
        throw Load_wider_value_to_narrowed_element((std::max)(sizeof(Ty), sizeof(RTy)), sizeof(Element_type));
    }

    template<typename RTy, typename Ty, typename Func,
        typename = typename std::enable_if_t<
            (sizeof(Element_type) >= sizeof(Ty) && sizeof(Element_type) >= sizeof(RTy))
            && std::is_assignable<std::function<RTy(Ty const &, Ty const &)>, Func>::value>
    >
    static
    auto
    adapter2(Func &&func)
    {
        return
            [&func](Element_type const &x, Element_type const &y)->Element_type {
                return
                    to_element(func(
                        reinterpret_cast<Ty const &>(x),
                        reinterpret_cast<Ty const &>(y));
            };
    }

    template<typename RTy, typename Ty, typename Func>
    static
    typename std::enable_if<
        (sizeof(Element_type) < sizeof(Ty) || sizeof(Element_type) < sizeof(RTy))
        && std::is_assignable<std::function<RTy(Ty const &, Ty const &, Ty const &)>, Func>::value,
        std::function<Element_type(Element_type const &, Element_type const &, Element_type const &)>
    >::type
    adapter3(Func &&)
    {
        throw Load_wider_value_to_narrowed_element((std::max)(sizeof(Ty), sizeof(RTy)), sizeof(Element_type));
    }

    template<typename RTy, typename Ty, typename Func,
        typename = typename std::enable_if_t<
            (sizeof(Element_type) >= sizeof(Ty) && sizeof(Element_type) >= sizeof(RTy))
            && std::is_assignable<std::function<RTy(Ty const &, Ty const &, Ty const &)>, Func>::value>
    >
    static
    auto
    adapter3(Func &&func)
    {
        return
            [&func](Element_type const &x, Element_type const &y, Element_type const &z)->Element_type {
                return to_element(func(
                    reinterpret_cast<Ty const &>(x),
                    reinterpret_cast<Ty const &>(y),
                    reinterpret_cast<Ty const &>(z)));
            };
    }
#endif

    static Element_type sll(Element_type const &x, Element_type const &y)
    {
        return x << y;
    }

    static Element_type sra(Element_type const &x, Element_type const &y)
    {
        return x >> y;
    }

    static Element_type srl(Element_type const &x, Element_type const &y)
    {
        return to_element(static_cast<typename std::make_unsigned<Element_type>::type>(x) >> y);
    }

    void
    vadd_vv(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) final
    {
        this->iterate(V_unit::instance(), std::plus<Element_type>(), vd, vs1, vs2, mode);
    }

    void
    vadd_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) final
    {
        using namespace std::placeholders;
        this->iterate(V_unit::instance(), std::bind(std::plus<Element_type>(), _1, Element_type(rs1)), vd, vs2, mode);
    }

    void
    vadd_vi(vreg_no vd, vreg_no vs2, int16_t imm, vop_type mode = vop_type::thread_all) final
    {
        using namespace std::placeholders;
        this->iterate(V_unit::instance(), std::bind(std::plus<Element_type>(), _1, Element_type(imm)), vd, vs2, mode);
    }

    void
    vsub_vv(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) final
    {
        this->iterate(V_unit::instance(), std::minus<Element_type>(), vd, vs1, vs2, mode);
    }

    void
    vsub_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) final
    {
        using namespace std::placeholders;
        this->iterate(V_unit::instance(), std::bind(std::minus<Element_type>(), _1, Element_type(rs1)), vd, vs2, mode);
    }

    void
    vmsle_vv(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) final
    {
        this->iterate_vm(V_unit::instance(),
                         std::less_equal<Element_type>(),
                         vd,
                         vs1,
                         vs2,
                         mode);
    }

    void
    vmsle_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) final
    {
        using namespace std::placeholders;
        this->iterate_vm(V_unit::instance(),
                         std::bind(std::less_equal<Element_type>(), _1, Element_type(rs1)),
                         vd,
                         vs2,
                         mode);
    }

    void
    vmsle_vi(vreg_no vd, vreg_no vs2, int16_t imm, vop_type mode = vop_type::thread_all) final
    {
        using namespace std::placeholders;
        this->iterate_vm(V_unit::instance(),
                         std::bind(std::less_equal<Element_type>(), _1, Element_type(imm)),
                         vd,
                         vs2,
                         mode);
    }

    void
    vmand_mm(vreg_no vd, vreg_no vs2, vreg_no vs1) final
    {
        this->iterate_mm(V_unit::instance(), std::logical_and<bool>(), vd, vs1, vs2);
    }

    void
    vmnand_mm(vreg_no vd, vreg_no vs2, vreg_no vs1) final
    {
        auto op = [](bool const& x, bool const& y)->bool {
            return !(x && y);
        };
        this->iterate_mm(V_unit::instance(), op, vd, vs1, vs2);
    }

    void
    vmnot_m(vreg_no vd, vreg_no vs1) final
    {
        vmnand_mm(vd, vs1, vs1);
    }

    void
    vmv_v_v(vreg_no vd, vreg_no vs1) final
    {
        auto op = [](Element_type const& x)->Element_type {
            return x;
        };
        this->iterate(V_unit::instance(), op, vd, vs1, vop_type::thread_all);
    }

    void
    vmv_v_x(vreg_no vd, xreg_type rs1) final
    {
        auto op = [&rs1]()->Element_type {
            return Element_type(rs1);
        };
        this->iterate(V_unit::instance(), op, vd);
    }

    void
    vmv_v_i(vreg_no vd, int16_t imm) final
    {
        auto op = [&imm]()->Element_type {
            return Element_type(imm);
        };
        this->iterate(V_unit::instance(), op, vd);
    }

#if 0
    void vsll(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(sll, vd, vs1, vs2);
    }

    void vsra(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(sra, vd, vs1, vs2);
    }

    void vsrl(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(srl, vd, vs1, vs2);
    }

    void vand(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::bit_and<Element_type>(), vd, vs1, vs2);
    }

    void vor(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::bit_or<Element_type>(), vd, vs1, vs2);
    }

    void vxor(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::bit_xor<Element_type>(), vd, vs1, vs2);
    }

    void vmul(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::multiplies<Element_type>(), vd, vs1, vs2);
    }

    void vmulh(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        auto op = [](Element_type const& x, Element_type const& y)->Element_type {
            return to_element(mulh(x, y));
        };
        this->iterate(op, vd, vs1, vs2);
    }

    void vmulhsu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        auto op = [](Element_type const& x, Element_type const& y)->Element_type {
            return to_element(mulhsu(x, y));
        };
        this->iterate(op, vd, vs1, vs2);
    }

    void vmulhu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        auto op = [](Element_type const& x, Element_type const& y)->Element_type {
            return to_element(mulhu(x, y));
        };
        this->iterate(op, vd, vs1, vs2);
    }

    void vdiv(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::divides<Element_type>(), vd, vs1, vs2);
    }

    void vdivu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        static auto const op = [](Element_type x, Element_type y)->Element_type {
            typedef typename std::make_unsigned<Element_type>::type uns_type;
            return static_cast<uns_type>(x) / static_cast<uns_type>(y);
        };
        this->iterate(op, vd, vs1, vs2);
    }

    void vrem(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(std::modulus<Element_type>(), vd, vs1, vs2);
    }

    void vremu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        static auto const op = [](Element_type x, Element_type y)->Element_type {
            typedef typename std::make_unsigned<Element_type>::type uns_type;
            return static_cast<uns_type>(x) % static_cast<uns_type>(y);
        };
        this->iterate(op, vd, vs1, vs2);
    }

    void vseq(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<bool, Element_type>(std::equal_to<Element_type>()), vd, vs1, vs2);
    }

    void vslt(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<bool, Element_type>(std::less<Element_type>()), vd, vs1, vs2);
    }

    void vsge(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<bool, Element_type>(std::greater_equal<Element_type>()), vd, vs1, vs2);
    }

    void vsltu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        static auto const op = [](Element_type x, Element_type y)->bool {
            typedef typename std::make_unsigned<Element_type>::type uns_type;
            return std::less<uns_type>()(static_cast<uns_type>(x), static_cast<uns_type>(y));
        };
        this->iterate(adapter2<bool, Element_type>(op), vd, vs1, vs2);
    }

    void vsgeu(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        static auto const op = [](Element_type x, Element_type y)->bool {
            typedef typename std::make_unsigned<Element_type>::type uns_type;
            return std::greater<uns_type>()(static_cast<uns_type>(x), static_cast<uns_type>(y));
        };
        this->iterate(adapter2<bool, Element_type>(op), vd, vs1, vs2);
    }

    void vslli(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(sll, _1, Element_type(imm)), vd, vs1);
    }

    void vsrli(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(srl, _1, Element_type(imm)), vd, vs1);
    }

    void vsrai(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(sra, _1, Element_type(imm)), vd, vs1);
    }

    void vandi(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(std::bit_and<Element_type>(), _1, Element_type(imm)),
                      vd, vs1);
    }

    void vori(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(std::bit_or<Element_type >(), _1, Element_type(imm)),
                      vd, vs1);
    }

    void vxori(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        using namespace std::placeholders;
        this->iterate(std::bind(std::bit_xor<Element_type >(), _1, Element_type(imm)),
                      vd, vs1);
    }

    void vfadd_w(vreg_no vd, vreg_no vs1, vreg_no vs2)final
    {
        this->iterate(adapter2<float32_t, float32_t>(std::plus<float32_t>()), vd, vs1, vs2);
    }

    void vfadd_d(vreg_no vd, vreg_no vs1, vreg_no vs2)final
    {
        this->iterate(adapter2<float64_t, float64_t>(std::plus<float64_t>()), vd, vs1, vs2);
    }

    void vfsub_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(std::minus<float32_t>()), vd, vs1, vs2);
    }

    void vfsub_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(std::minus<float64_t>()), vd, vs1, vs2);
    }

    void vfmul_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(std::multiplies<float32_t>()), vd, vs1, vs2);
    }

    void vfmul_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(std::multiplies<float64_t>()), vd, vs1, vs2);
    }

    void vfdiv_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(std::divides<float32_t>()), vd, vs1, vs2);
    }

    void vfdiv_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(std::divides<float64_t>()), vd, vs1, vs2);
    }

    void vfsgnj_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(fsgnj<float32_t>), vd, vs1, vs2);
    }

    void vfsgnj_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(fsgnj<float64_t>), vd, vs1, vs2);
    }

    void vfsgnjn_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(fsgnjn<float32_t>), vd, vs1, vs2);
    }

    void vfsgnjn_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(fsgnjn<float64_t>), vd, vs1, vs2);
    }

    void vfsgnjx_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float32_t, float32_t>(fsgnjx<float32_t>), vd, vs1, vs2);
    }

    void vfsgnjx_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        this->iterate(adapter2<float64_t, float64_t>(fsgnjx<float64_t>), vd, vs1, vs2);
    }

    void vfmadd_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float32_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type, curr_type) = std::fmaf;
        this->iterate(adapter3<curr_type, curr_type>(fn), vd, vs1, vs2, vs3);
    }

    void vfmadd_d(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float64_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type, curr_type) = std::fma;
        this->iterate(adapter3<curr_type, curr_type>(fn), vd, vs1, vs2, vs3);
    }

    void vfmsub_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float32_t curr_type;
        static auto const op = [](curr_type const & x, curr_type const & y, curr_type const & z)->curr_type {
            return std::fmaf(x, y, -z);
        };
        this->iterate(adapter3<curr_type, curr_type>(op), vd, vs1, vs2, vs3);
    }

    void vfmsub_d(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float64_t curr_type;
        static auto const op = [](curr_type const & x, curr_type const & y, curr_type const & z)->curr_type {
            return std::fma(x, y, -z);
        };
        this->iterate(adapter3<curr_type, curr_type>(op), vd, vs1, vs2, vs3);
    }

    void vfmaddwdn_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float32_t small_type;
        typedef float32_t big_type;
        static auto const op = [](small_type const & x, small_type const & y, small_type const & z)->big_type {
            return std::fma(big_type(x), big_type(y), big_type(z));
        };
        this->iterate(adapter3<big_type, small_type>(op), vd, vs1, vs2, vs3);
    }

    void vfmsubwdn_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) final
    {
        typedef float32_t small_type;
        typedef float32_t big_type;
        static auto const op = [](small_type const & x, small_type const & y, small_type const & z)->big_type {
            return std::fma(big_type(x), big_type(y), big_type(-z));
        };
        this->iterate(adapter3<big_type, small_type>(op), vd, vs1, vs2, vs3);
    }

    void vfmin_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float32_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type) = std::fminf;
        this->iterate(adapter2<curr_type, curr_type>(fn), vd, vs1, vs2);
    }

    void vfmin_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float64_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type) = std::fmin;
        this->iterate(adapter2<curr_type, curr_type>(fn), vd, vs1, vs2);
    }

    void vfmax_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float32_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type) = std::fmaxf;
        this->iterate(adapter2<curr_type, curr_type>(fn), vd, vs1, vs2);
    }

    void vfmax_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float64_t curr_type;
        static curr_type(*const fn)(curr_type, curr_type) = std::fmax;
        this->iterate(adapter2<curr_type, curr_type>(fn), vd, vs1, vs2);
    }

    void vfsqrt_w(vreg_no vd, vreg_no vs1) final
    {
        typedef float32_t curr_type;
        static curr_type(*const fn)(curr_type) = std::sqrt;
        this->iterate(adapter1<curr_type, curr_type>(fn), vd, vs1);
    }

    void vfsqrt_d(vreg_no vd, vreg_no vs1) final
    {
        typedef float64_t curr_type;
        static curr_type(*const fn)(curr_type) = std::sqrt;
        this->iterate(adapter1<curr_type, curr_type>(fn), vd, vs1);
    }

    void vfeq_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float32_t curr_type;
        this->iterate(adapter2<bool, curr_type>(std::equal_to<curr_type>()), vd, vs1, vs2);
    }

    void vfeq_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float64_t curr_type;
        this->iterate(adapter2<bool, curr_type>(std::equal_to<curr_type>()), vd, vs1, vs2);
    }

    void vflt_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float32_t curr_type;
        static bool(*const fn)(curr_type, curr_type) = std::isless;
        this->iterate(adapter2<bool, curr_type>(fn), vd, vs1, vs2);
    }

    void vflt_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float64_t curr_type;
        static bool(*const fn)(curr_type, curr_type) = std::isless;
        this->iterate(adapter2<bool, curr_type>(fn), vd, vs1, vs2);
    }

    void vfle_w(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float32_t curr_type;
        static bool(*const fn)(curr_type, curr_type) = std::islessequal;
        this->iterate(adapter2<bool, curr_type>(fn), vd, vs1, vs2);
    }

    void vfle_d(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef float64_t curr_type;
        static bool(*const fn)(curr_type, curr_type) = std::islessequal;
        this->iterate(adapter2<bool, curr_type>(fn), vd, vs1, vs2);
    }

    void vaddw(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef int32_t curr_type;
        this->iterate(adapter2<curr_type, curr_type>(std::plus<curr_type>()), vd, vs1, vs2);
    }

    void vsubw(vreg_no vd, vreg_no vs1, vreg_no vs2) final
    {
        typedef int32_t curr_type;
        this->iterate(adapter2<curr_type, curr_type>(std::minus<curr_type>()), vd, vs1, vs2);
    }

    void vaddwi(vreg_no vd, vreg_no vs1, int16_t imm) final
    {
        typedef int32_t curr_type;
        using namespace std::placeholders;
        this->iterate(adapter1<curr_type, curr_type>(std::bind(std::plus<curr_type>(), _1, curr_type(imm))), vd, vs1);
    }
#endif
};

template<typename Element_type>
class Float_operations_impl
    : virtual public Float_operations
    , Float_operations_essentials<Element_type>
{
private:
    virtual operator Float_operations &() final
    {
        return *this;
    }

    void
    vfmacc_vf(vreg_no vd, float rs1, vreg_no vs2, vop_type mode = vop_type::thread_all) final
    {
        auto op = [&rs1](Element_type const& x, Element_type const& y)->Element_type {
            return rs1 * x + y;
        };
        this->iterate(V_unit::instance(), op, vd, vs2, vd, mode);
    }
};

Operations_impl<int8_t> op8;
Operations_impl<int16_t> op16;
Operations_impl<int32_t> op32;
Operations_impl<int64_t> op64;

Float_operations_impl<float32_t> fop32;
Float_operations_impl<float64_t> fop64;

static thread_local std::unique_ptr<State> p_state;

static inline constexpr size_t
bits(size_t _value, size_t _offset = 0)
{
    return 0 == _value ? _offset : bits(_value >> 1, _offset + 1);
}
}  // namespace

class State_impl
    : virtual protected State
{
    typedef State_impl This_class;
protected:
    State_impl()
        : m_vstart(0)
        , m_vl(0)
        , m_mul(m1)
        , m_mask_reg(v0)
    {
    }

private:
    size_t setvstart(size_t vstart)final
    {
        return this->m_vstart = vstart;
    }

    size_t setvl(size_t vl)final
    {
        return this->m_vl = vl;
    }

    void setill(bool ill)final
    {
        this->m_ill = ill;
    }

    void setew(size_t ew)final
    {
        switch(ew) {
            case 0b000:
                this->m_op_performer = &op8;
                this->m_fop_performer = nullptr;
                break;
            case 0b001:
                this->m_op_performer = &op16;
                this->m_fop_performer = nullptr;
                break;
            case 0b010:
                this->m_op_performer = &op32;
                this->m_fop_performer = &fop32;
                break;
            case 0b011:
                this->m_op_performer = &op64;
                this->m_fop_performer = &fop64;
                break;
        }
        this->m_ew = ew;
    }

    void setmul(size_t mul)final
    {
        this->m_mul = mul;
    }

    size_t vstart()const final
    {
        return m_vstart;
    }

    size_t vl()const final
    {
        return m_vl;
    }

    size_t sew()const final
    {
        return 8 << m_ew;
    }

    size_t lmul()const final
    {
        return 1 << m_mul;
    }

    bool is_ill()const final
    {
        return m_ill;
    }

    size_t vlmax()const final
    {
        return m_mul * implementation::V_unit::VLEN / sew();
    }

    inline bool is_enabled(size_t i)const
    {
        return 0 != this->mask_bit(i);
    }

    bool
    is_valid_reg(vreg_no _reg)const final
    {
        return (_reg % lmul()) == 0;
    }

    char *
    elt_ptr(vreg_no _reg, size_t _ind) final
    {
        if (!is_valid_reg(_reg)) {
            throw Register_out_of_config_range(_reg);
        }
        size_t skip_rows = _reg;

        size_t elements_in_stripe = V_unit::SLEN / sew();
        size_t elements_in_group = elements_in_stripe * lmul();

        size_t num_of_group = _ind / elements_in_group;

        size_t row = (_ind % elements_in_group) / elements_in_stripe;
        size_t col = _ind % elements_in_stripe;

        size_t bits = (skip_rows + row) * V_unit::VLEN + (num_of_group * elements_in_stripe + col) * sew();
        return &m_register_file[bits / 8];
    }

    char const *
    elt_ptr(vreg_no _reg, size_t _ind)const final
    {
        return const_cast<This_class*>(this)->elt_ptr(_reg, _ind);
    }

    void
    set_mask(vreg_no _reg, size_t _ind, bool value) final
    {
        size_t mlen = sew() / lmul();
        size_t byte_num = _reg * V_unit::VLEN / 8 + (_ind * mlen) / 8;
        char *byte_ptr = &m_register_file[byte_num];

        // zeroing mlen bits
        size_t bits_left = mlen;
        char *ptr = byte_ptr;
        while (bits_left >= 8) {
            *ptr = 0;
            bits_left -= 8;
            ++ptr;
        }
        if (bits_left) {
            *ptr = *ptr & ~((1 << bits_left) - 1);
        }

        // set LSB to the value
        *byte_ptr = *byte_ptr & ~(1u) | !!value;
    }

    bool
    get_mask(vreg_no _reg, size_t _ind)const final
    {
        size_t mlen = sew() / lmul();
        size_t byte_num = _reg * V_unit::VLEN / 8 + (_ind * mlen) / 8;
        char const *byte_ptr = &m_register_file[byte_num];

        return 0 != (*byte_ptr & (1 << (mlen % 8)));
    }

    bool
    mask_bit(size_t i)const final
    {
        return get_mask(v0, i);
    }

    Operations&
    get_op_performer()const final
    {
        return *m_op_performer;
    }

    Float_operations&
    get_fop_performer()const final
    {
        return *m_fop_performer;
    }

private:
    std::array<char, V_unit::VLEN / 8 * V_unit::NREGS> m_register_file;

    Operations* m_op_performer;
    Float_operations* m_fop_performer;

    size_t m_vstart;
    size_t m_vl;
    bool m_ill;
    size_t m_ew;
    size_t m_mul;
    vreg_no m_mask_reg;
};

namespace {
class V_unit_impl
    : State_impl
{
    V_unit_impl()
        : State_impl()
    {}

public:
    static void
        init()
    {
        p_state.reset(new V_unit_impl());
    }
};

}  // namespace

V_unit&
V_unit::instance()
{
    return *p_state;
}

}  // namespace implementation

size_t
vsetvl(size_t _avl, size_t _vtype)
{
    using implementation::p_state;

    if (!p_state) {
        implementation::V_unit_impl::init();
    }

    bool const ill = 0 != ((_vtype >> (sizeof(xreg_type) - 1)) && 0b1);
    size_t mul = 0;
    size_t ew = 0;
    size_t avl = 0;
    if (!ill) {
        mul = _vtype & 0b11;
        ew = (_vtype >> 2) & 0b111;
        size_t lmul = 1 << mul;
        size_t sew = 8 << ew;

        if (_avl > 0) {
            size_t vlmax = lmul * implementation::V_unit::VLEN / sew;

            if (_avl <= vlmax) {
                avl = _avl;
            } else if (_avl >= 2 * vlmax) {
                avl = vlmax;
            } else {
                avl = (_avl + 1) / 2;
            }
        }
    }

    p_state->setill(ill);
    p_state->setmul(mul);
    p_state->setew(ew);
    p_state->setvstart(0);
    return p_state->setvl(avl);
}

size_t
vsetvli(size_t _avl, int16_t _vtypei)
{
    return vsetvl(_avl, size_t(_vtypei));
}

#if 0
// mask is always register 0
void
vsetmask(vreg_no _size)
{
    using implementation::p_state;

    p_state->set_mask_reg(_size);
}
#endif

}  // namespace spec_0_7
}  // namespace v
}  // namespace riscv
