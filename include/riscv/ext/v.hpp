/**
    @file v.hpp
    @copyright ©2019 Syntacore.
    @authors
        Grigory Okhotnikov <go@syntacore.com>
    @brief Vector extension simulator (v0.7)
*/

#ifndef RISCV_EXT_V_HPP_
#define RISCV_EXT_V_HPP_

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>

#define PASTE_(OP1, OP2) OP1 ## OP2
#define CONCAT_(OP1, OP2) PASTE_(OP1, OP2)

namespace riscv {

typedef int64_t xreg_type;

class Invalid_instruction
    : public std::invalid_argument
{
protected:
    Invalid_instruction(std::string const& msg)
        : std::invalid_argument(std::string("exception: invalid instruction: ") + msg)
    {}

};

namespace v {
inline namespace spec_0_7 {

enum class vop_type : uint8_t
{
    thread_all = 0b0,
    masked_in = 0b1,
};

enum vreg_no : uint8_t
{
    v0 = 0,
    v1 = 1,
    v2 = 2,
    v3 = 3,
    v4 = 4,
    v5 = 5,
    v6 = 6,
    v7 = 7,
    v8 = 8,
    v9 = 9,
    v10 = 10,
    v11 = 11,
    v12 = 12,
    v13 = 13,
    v14 = 14,
    v15 = 15,
    v16 = 16,
    v17 = 17,
    v18 = 18,
    v19 = 19,
    v20 = 20,
    v21 = 21,
    v22 = 22,
    v23 = 23,
    v24 = 24,
    v25 = 25,
    v26 = 26,
    v27 = 27,
    v28 = 28,
    v29 = 29,
    v30 = 30,
    v31 = 31
};

enum vreg_ew : uint8_t
{
    e8 = 0b00,
    e16 = 0b01,
    e32 = 0b10,
    e64 = 0b11,
    e128 = 0b100
};

enum vreg_mul : uint8_t
{
    m1 = 0b00,
    m2 = 0b01,
    m4 = 0b10,
    m8 = 0b11
};

inline size_t
vtype(vreg_ew ew, vreg_mul mul = m1, size_t vill = 0)
{
    return vill << (sizeof(xreg_type) - 1) | ew << 2 | mul;
}

inline int16_t
vtypei(vreg_ew ew, vreg_mul mul = m1)
{
    return (ew << 2) | mul;
}

namespace implementation {

typedef float float32_t;
typedef double float64_t;

#if 0
class Bad_element_size
    : public Invalid_instruction
{
public:
    Bad_element_size(size_t el_size)
        : Invalid_instruction(std::string("RVV config bad element size: ") + std::to_string(el_size))
    {}
};
#endif

class State_not_configured
    : public Invalid_instruction
{
public:
    State_not_configured()
        : Invalid_instruction(std::string("Illegal configuration (vill is set)"))
    {}
};

class Register_out_of_config_range
    : public Invalid_instruction
{
public:
    Register_out_of_config_range(vreg_no reg)
        : Invalid_instruction(std::string("RVV register is out of config range : ") + std::to_string(static_cast<unsigned>(reg)))
    {}
};

class Load_wider_value_to_narrowed_element
    : public Invalid_instruction
{
public:
    Load_wider_value_to_narrowed_element(size_t value_size, size_t el_size)
        : Invalid_instruction(std::string("RVV load wider value (size=") + std::to_string(value_size) + ") to narrowed element (size" + std::to_string(el_size) + ")")
    {}
};

class Instruction_undefined_for_element_size
    : public Invalid_instruction
{
public:
    Instruction_undefined_for_element_size(size_t el_size)
        : Invalid_instruction(std::string("Instruction undefined for element size ") + std::to_string(el_size))
    {}
};

class V_unit;
class State;

template<size_t N>
struct Size_traits;

template<typename Memory_type>
class Loader
{
private:
    Loader(Loader const&) = delete;
    Loader& operator = (Loader const&) = delete;

protected:
    Loader() = default;
    virtual ~Loader() = default;

protected:
    static Memory_type
    to_element(bool const &_val)
    {
        return Memory_type(!!_val);
    }

    template<typename Value_type>
    static typename std::enable_if<
        (sizeof(Value_type) > sizeof(Memory_type)),
        Memory_type
    >::type
    to_element(Value_type const &)
    {
        throw Load_wider_value_to_narrowed_element(sizeof(Value_type), sizeof(Memory_type));
    }

    template<typename Value_type>
    static typename std::enable_if<
        sizeof(Memory_type) == sizeof(Value_type),
        Memory_type
    >::type
    to_element(Value_type const &_val)
    {
        return reinterpret_cast<Memory_type const &>(_val);
    }

    template<typename Value_type>
    static typename std::enable_if<
        (sizeof(Value_type) < sizeof(Memory_type)) &&
        std::is_integral<Value_type>::value &&
        std::is_unsigned<Value_type>::value,
        Memory_type
    >::type
    to_element(Value_type const &_val)
    {
        return static_cast<Memory_type>(static_cast<typename std::make_unsigned<Memory_type>::type>(_val));
    }

    template<typename Value_type>
    static typename std::enable_if<
        (sizeof(Value_type) < sizeof(Memory_type)) &&
        std::is_integral<Value_type>::value &&
        std::is_signed<Value_type>::value,
        Memory_type
    >::type
    to_element(Value_type const &_val)
    {
        return static_cast<Memory_type>(_val);
    }

    template<typename Value_type>
    static typename std::enable_if<
        (sizeof(Value_type) < sizeof(Memory_type)) &&
        std::is_floating_point<Value_type>::value,
        Memory_type
    >::type
    to_element(Value_type const &_val)
    {
        typedef typename std::make_unsigned<Memory_type>::type uel_type;
        //        static uel_type const NaN_box = ~uel_type(0) << (CHAR_BIT * sizeof(Value_type));

        typedef typename Size_traits<sizeof(Value_type)>::uint_type uval_type;
        //        return static_cast<Memory_type>(NaN_box | uel_type(reinterpret_cast<uval_type const&>(_val)));
        return static_cast<Memory_type>(uel_type(reinterpret_cast<uval_type const &>(_val)));
    }

public:
    virtual void operator()(V_unit& vu, vreg_no vd, Memory_type const* rs1, ptrdiff_t rs2, vop_type mode) = 0;
    virtual void operator()(V_unit& vu, vreg_no vd, Memory_type const* rs1, vreg_no idx, vop_type mode) = 0;
};

template<typename Memory_type>
class Saver
{
private:
    Saver(Saver const&) = delete;
    Saver& operator = (Saver const&) = delete;

protected:
    Saver() = default;
    virtual ~Saver() = default;

public:
    virtual void operator()(V_unit& vu, vreg_no vs1, Memory_type* rs1, ptrdiff_t rs2, vop_type mode) const = 0;
    virtual void operator()(V_unit& vu, vreg_no vs1, Memory_type* rs1, vreg_no idx, vop_type mode) const = 0;
};

class Operations
{
private:
    Operations(Operations const&) = delete;
    Operations& operator = (Operations const&) = delete;

protected:
    Operations() = default;
    virtual ~Operations() = default;

public:
    virtual operator Loader<int8_t>& () = 0;
    virtual operator Loader<int16_t>& () = 0;
    virtual operator Loader<int32_t>& () = 0;
    virtual operator Loader<int64_t>& () = 0;

    virtual operator Loader<uint8_t>& () = 0;
    virtual operator Loader<uint16_t>& () = 0;
    virtual operator Loader<uint32_t>& () = 0;

    virtual operator Saver<int8_t>& () = 0;
    virtual operator Saver<int16_t>& () = 0;
    virtual operator Saver<int32_t>& () = 0;
    virtual operator Saver<int64_t>& () = 0;

    virtual void vadd_vv(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) = 0;
    virtual void vadd_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) = 0;
    virtual void vadd_vi(vreg_no vd, vreg_no vs2, int16_t imm, vop_type mode = vop_type::thread_all) = 0;

    virtual void vsub_vv(vreg_no vd, vreg_no vs1, vreg_no vs2, vop_type mode = vop_type::thread_all) = 0;
    virtual void vsub_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) = 0;

    virtual void vmsle_vv(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) = 0;
    virtual void vmsle_vx(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) = 0;
    virtual void vmsle_vi(vreg_no vd, vreg_no vs2, int16_t imm, vop_type mode = vop_type::thread_all) = 0;

    virtual void vmand_mm(vreg_no vd, vreg_no vs2, vreg_no vs1) = 0;
    virtual void vmnand_mm(vreg_no vd, vreg_no vs2, vreg_no vs1) = 0;

    virtual void vmnot_m(vreg_no vd, vreg_no vs1) = 0;

    virtual void vmv_v_v(vreg_no vd, vreg_no vs1) = 0;
    virtual void vmv_v_x(vreg_no vd, xreg_type rs1) = 0;
    virtual void vmv_v_i(vreg_no vd, int16_t imm) = 0;
#if 0
    virtual void vaddw(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsubw(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vsll(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsra(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsrl(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vand(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vor(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vxor(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vseq(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vslt(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsge(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsltu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vsgeu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vaddi(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vslli(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vsrli(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vsrai(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vandi(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vori(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vxori(vreg_no vd, vreg_no vs1, int16_t imm) = 0;
    virtual void vaddwi(vreg_no vd, vreg_no vs1, int16_t imm) = 0;

    virtual void vmul(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vmulh(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vmulhsu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vmulhu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vdiv(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vdivu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vrem(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vremu(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vfadd_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfadd_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsub_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsub_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfmul_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfmul_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfdiv_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfdiv_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vfsgnj_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsgnj_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsgnjn_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsgnjn_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsgnjx_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsgnjx_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;

    virtual void vfmadd_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;
    virtual void vfmadd_d(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;
    virtual void vfmsub_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;
    virtual void vfmsub_d(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;
    virtual void vfmaddwdn_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;
    virtual void vfmsubwdn_w(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3) = 0;

    virtual void vfmin_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfmin_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfmax_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfmax_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfeq_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfeq_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vflt_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vflt_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfle_w(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfle_d(vreg_no vd, vreg_no vs1, vreg_no vs2) = 0;
    virtual void vfsqrt_w(vreg_no vd, vreg_no vs1) = 0;
    virtual void vfsqrt_d(vreg_no vd, vreg_no vs1) = 0;

    virtual void vinsx(vreg_no vd, int32_t value, size_t idx) = 0;
    virtual void vmiota(vreg_no vd) = 0;
#endif
};

class Float_operations
{
private:
    Float_operations(Float_operations const&) = delete;
    Float_operations& operator = (Float_operations const&) = delete;

protected:
    Float_operations() = default;
    virtual ~Float_operations() = default;

public:
    virtual void vfmacc_vf(vreg_no vd, float rs1, vreg_no vs2, vop_type mode = vop_type::thread_all) = 0;
};

#ifndef RVV_ELEN
#define RVV_ELEN 64
#endif

#ifndef RVV_VLEN
#define RVV_VLEN 256
#endif

#ifndef RVV_SLEN
#define RVV_SLEN 64
#endif

class V_unit
{
private:
    V_unit(V_unit const&) = delete;
    V_unit& operator=(V_unit const&) = delete;

protected:
    V_unit() = default;
    virtual ~V_unit() = default;

public:
    static size_t const ELEN = RVV_ELEN;
    static size_t const VLEN = RVV_VLEN;
    static size_t const SLEN = RVV_SLEN;
    static size_t const NREGS = 32;

    static V_unit& instance();

    virtual Operations& get_op_performer()const = 0;
    virtual Float_operations& get_fop_performer()const = 0;
};

template<typename Ty>
inline void
load(vreg_no vd, Ty const* rs1, ptrdiff_t rs2 = sizeof(Ty), vop_type mode = vop_type::thread_all)
{
    static_cast<Loader<Ty>&>(static_cast<Operations&>(V_unit::instance().get_op_performer()))(V_unit::instance(), vd, rs1, rs2, mode);
}

template<typename Ty>
inline void
load(vreg_no vd, Ty const* rs1, vreg_no vs1, vop_type mode = vop_type::thread_all)
{
    static_cast<Loader<Ty>&>(static_cast<Operations&>(V_unit::instance().get_op_performer()))(V_unit::instance(), vd, rs1, vs1, mode);
}

template<typename Ty>
inline void
save(vreg_no vs1, Ty* rs1, ptrdiff_t rs2 = sizeof(Ty), vop_type mode = vop_type::thread_all)
{
    static_cast<Saver<Ty>&>(static_cast<Operations&>(V_unit::instance().get_op_performer()))(V_unit::instance(), vs1, rs1, rs2, mode);
}

template<typename Ty>
inline void
save(vreg_no vs1, Ty* rs1, vreg_no vs2, vop_type mode = vop_type::thread_all)
{
    static_cast<Saver<Ty>&>(static_cast<Operations&>(V_unit::instance().get_op_performer()))(V_unit::instance(), vs1, rs1, vs2, mode);
}

}  // namespace implementation

size_t
vsetvl(size_t, size_t);

size_t
vsetvli(size_t, int16_t);

#define DEF_B_D(INNER_DEF_) INNER_DEF_(b,8) INNER_DEF_(h,16) INNER_DEF_(w,32)

/// Load constant-stride instructions
///@{

#define DEF_LOAD_CONSTANT_STRIDE_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vd, TYPE const* rs1, ptrdiff_t rs2, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        load(vd, rs1, rs2, mode); \
    }

/// Load as signed integer (sign extended)
///@{

#define DEF_LOAD_INT_CONSTANT_STRIDE_(CHR,NUM) DEF_LOAD_CONSTANT_STRIDE_(CONCAT_(vls,CHR), CONCAT_(int,CONCAT_(NUM,_t)))
DEF_B_D(DEF_LOAD_INT_CONSTANT_STRIDE_)
#undef DEF_LOAD_INT_CONSTANT_STRIDE_
///@}

/// Load as unsigned integer (zero extended)
///@{
#define DEF_LOAD_UNSIGNED_CONSTANT_STRIDE_(CHR,NUM) DEF_LOAD_CONSTANT_STRIDE_(CONCAT_(CONCAT_(vls,CHR),u), CONCAT_(CONCAT_(uint,NUM),_t))
DEF_B_D(DEF_LOAD_UNSIGNED_CONSTANT_STRIDE_)
#undef DEF_LOAD_UNSIGNED_CONSTANT_STRIDE_
///@}

#undef DEF_LOAD_CONSTANT_STRIDE_

///@}

/// Load unit-stride instructions
///@{

#define DEF_LOAD_UNIT_STRIDE_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vd, TYPE const rs1[], vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        load(vd, rs1, sizeof(TYPE), mode); \
    }

/// Load as signed integer (sign extended)
///@{
#define DEF_LOAD_INT_UNIT_STRIDE_(CHR,NUM) DEF_LOAD_UNIT_STRIDE_(CONCAT_(vl,CHR), CONCAT_(CONCAT_(int,NUM),_t))
DEF_B_D(DEF_LOAD_INT_UNIT_STRIDE_)
#undef DEF_LOAD_INT_UNIT_STRIDE_
///@}

/// Load as unsigned integer (zero extended)
///@{
#define DEF_LOAD_UNSIGNED_UNIT_STRIDE_(CHR,NUM) DEF_LOAD_UNIT_STRIDE_(CONCAT_(CONCAT_(vl,CHR),u), CONCAT_(CONCAT_(uint,NUM),_t))
DEF_B_D(DEF_LOAD_UNSIGNED_UNIT_STRIDE_)
#undef DEF_LOAD_UNSIGNED_UNIT_STRIDE_
///@}

#undef DEF_LOAD_UNIT_STRIDE_
///@}

/// Load indexed (scatter-gather)
///@{
#define DEF_LOAD_INDEXED_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vd, TYPE const* rs1, vreg_no vs1, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        load(vd, rs1, vs1, mode); \
    }

/// Load as signed integer (sign extended)
///@{
#define DEF_LOAD_INT_INDEXED_(CHR,NUM) DEF_LOAD_INDEXED_(CONCAT_(vlx,CHR), CONCAT_(CONCAT_(int,NUM),_t))
DEF_B_D(DEF_LOAD_INT_INDEXED_)
#undef DEF_LOAD_INT_INDEXED_
///@}

/// Load as unsigned integer (zero extended)
///@{
#define DEF_LOAD_UNSIGNED_INDEXED_(CHR,NUM) DEF_LOAD_INDEXED_(CONCAT_(vlx,CONCAT_(CHR,u)), CONCAT_(CONCAT_(uint,NUM),_t))
DEF_B_D(DEF_LOAD_UNSIGNED_INDEXED_)
#undef DEF_LOAD_UNSIGNED_INDEXED_
///@}

///@}

/// Constant-stride store instructions
///@{
#define DEF_SAVE_CONSTANT_STRIDE_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vs1, TYPE* rs1, ptrdiff_t rs2, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        save(vs1, rs1, rs2, mode); \
    }

#define DEF_SAVE_INT_CONSTANT_STRIDE_(CHR,NUM) DEF_SAVE_CONSTANT_STRIDE_(CONCAT_(vss,CHR), CONCAT_(CONCAT_(int,NUM),_t))
DEF_B_D(DEF_SAVE_INT_CONSTANT_STRIDE_)
#undef DEF_SAVE_INT_CONSTANT_STRIDE_
#undef DEF_SAVE_CONSTANT_STRIDE_
///@}

/// Store unit-stride instructions
///@{
#define DEF_SAVE_UNIT_STRIDE_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vs1, TYPE rs1[], vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        save(vs1, rs1, sizeof(TYPE), mode); \
    }

#define DEF_SAVE_INT_UNIT_STRIDE_(CHR,NUM) DEF_SAVE_UNIT_STRIDE_(CONCAT_(vs,CHR), CONCAT_(CONCAT_(int,NUM),_t))
DEF_B_D(DEF_SAVE_INT_UNIT_STRIDE_)
#undef DEF_SAVE_INT_UNIT_STRIDE_
#undef DEF_SAVE_UNIT_STRIDE_
///@}

/// indexed-ordered store (scatter) instructions
///@{
#define DEF_SAVE_INDEXED_(NAME,TYPE) \
    inline void CONCAT_(NAME,_v)(vreg_no vs1, TYPE* rs1, vreg_no vs2, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        save(vs1, rs1, vs2, mode); \
    }

#define DEF_SAVE_INT_INDEXED_(CHR,NUM) DEF_SAVE_INDEXED_(CONCAT_(vsx,CHR), CONCAT_(CONCAT_(int,NUM),_t))
DEF_B_D(DEF_SAVE_INT_INDEXED_)
#undef DEF_SAVE_INT_INDEXED_
///@}

#undef DEF_SAVE_INDEXED_

#undef DEF_B_D
#undef DEF_B_W

#define DEF_BIN_OP_VV(NAM) \
    inline void CONCAT_(NAM,_vv)(vreg_no vd, vreg_no vs2, vreg_no vs1, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_vv)(vd, vs2, vs1, mode); \
    }

#define DEF_BIN_OP_VX(NAM) \
    inline void CONCAT_(NAM,_vx)(vreg_no vd, vreg_no vs2, xreg_type rs1, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_vx)(vd, vs2, rs1, mode); \
    }

#define DEF_BIN_OP_VI(NAM) \
    inline void CONCAT_(NAM,_vi)(vreg_no vd, vreg_no vs2, int16_t imm, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_vi)(vd, vs2, imm, mode); \
    }

#define DEF_BIN_OP_VXI(nam) \
    DEF_BIN_OP_VV(nam) \
    DEF_BIN_OP_VX(nam) \
    DEF_BIN_OP_VI(nam)

DEF_BIN_OP_VXI(vadd)

DEF_BIN_OP_VV(vsub)
DEF_BIN_OP_VX(vsub)

// DEF_BIN_OP_VXI(vmsle)
DEF_BIN_OP_VI(vmsle)

#define DEF_BIN_OP_MM(NAM) \
    inline void CONCAT_(NAM,_mm)(vreg_no vd, vreg_no vs2, vreg_no vs1) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_mm)(vd, vs2, vs1); \
    }

#define DEF_BIN_OP_M(NAM) \
    inline void CONCAT_(NAM,_m)(vreg_no vd, vreg_no vs1, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_m)(vd, vs1, mode); \
    }

#define DEF_BIN_OP_M_NO_MODE(NAM) \
    inline void CONCAT_(NAM,_m)(vreg_no vd, vreg_no vs1) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).CONCAT_(NAM,_m)(vd, vs1); \
    }

DEF_BIN_OP_MM(vmand)
DEF_BIN_OP_MM(vmnand)
DEF_BIN_OP_M_NO_MODE(vmnot)

#undef DEF_BIN_OP_M_NO_MODE
#undef DEF_BIN_OP_MM
#undef DEF_BIN_OP_M

#if 0
DEF_BIN_OP(vsll)
DEF_BIN_OP(vsra)
DEF_BIN_OP(vsrl)
DEF_BIN_OP(vand)
DEF_BIN_OP(vor)
DEF_BIN_OP(vxor)

DEF_BIN_OP(vseq)
DEF_BIN_OP(vslt)
DEF_BIN_OP(vsge)
DEF_BIN_OP(vsltu)
DEF_BIN_OP(vsgeu)
#endif

#define DEF_BIN_IMM_OP(NAM) \
    inline void NAM(vreg_no vd, vreg_no vs1, int16_t imm, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).NAM(vd, vs1, imm, mode); \
    }

#if 0
DEF_BIN_IMM_OP(vaddi)
DEF_BIN_IMM_OP(vslli)
DEF_BIN_IMM_OP(vsrli)
DEF_BIN_IMM_OP(vsrai)
DEF_BIN_IMM_OP(vandi)
DEF_BIN_IMM_OP(vori)
DEF_BIN_IMM_OP(vxori)

/// 32-bit operations
///@{
DEF_BIN_OP(vaddw)
DEF_BIN_OP(vsubw)
DEF_BIN_IMM_OP(vaddwi)
///@}

DEF_BIN_OP(vmul)
DEF_BIN_OP(vmulh)
DEF_BIN_OP(vmulhsu)
DEF_BIN_OP(vmulhu)

DEF_BIN_OP(vdiv)
DEF_BIN_OP(vdivu)
DEF_BIN_OP(vrem)
DEF_BIN_OP(vremu)
#endif

inline void vmv_v_v(vreg_no vd, vreg_no vs1)
{
    using namespace implementation;
    static_cast<Operations&>(V_unit::instance().get_op_performer()).vmv_v_v(vd, vs1);
}

inline void vmv_v_x(vreg_no vd, xreg_type rs1)
{
    using namespace implementation;
    static_cast<Operations&>(V_unit::instance().get_op_performer()).vmv_v_x(vd, rs1);
}

inline void vmv_v_i(vreg_no vd, int16_t imm)
{
    using namespace implementation;
    static_cast<Operations&>(V_unit::instance().get_op_performer()).vmv_v_i(vd, imm);
}

#if 0
template<vreg_no vd, vreg_no vs1, vreg_no vs2, vop_type mode = vop_type::thread_all> inline void vmulwdn();

/// Integer reduction operations
///@{
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vredsum();

template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vredmax();
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vredmaxu();

template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vredmin();
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vredminu();
///@}
#endif

// void vfmacc_vf(vreg_no vd, float rs1, vreg_no vs2, vop_type mode = vop_type::thread_all) = 0;

#define DEF_BIN_OP_VF(NAM) \
    inline void CONCAT_(NAM,_vf)(vreg_no vd, float rs1, vreg_no vs2, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Float_operations&>(V_unit::instance().get_fop_performer()).CONCAT_(NAM,_vf)(vd, rs1, vs2, mode); \
    }

DEF_BIN_OP_VF(vfmacc)

#if 0
DEF_BIN_OP(vfadd_w)
DEF_BIN_OP(vfadd_d)

DEF_BIN_OP(vfsub_w)
DEF_BIN_OP(vfsub_d)

DEF_BIN_OP(vfmul_w)
DEF_BIN_OP(vfmul_d)

DEF_BIN_OP(vfdiv_w)
DEF_BIN_OP(vfdiv_d)

DEF_BIN_OP(vfsgnj_w)
DEF_BIN_OP(vfsgnj_d)

DEF_BIN_OP(vfsgnjn_w)
DEF_BIN_OP(vfsgnjn_d)

DEF_BIN_OP(vfsgnjx_w)
DEF_BIN_OP(vfsgnjx_d)

DEF_BIN_OP(vfmin_w)
DEF_BIN_OP(vfmin_d)

DEF_BIN_OP(vfmax_w)
DEF_BIN_OP(vfmax_d)

DEF_BIN_OP(vfeq_w)
DEF_BIN_OP(vfeq_d)

DEF_BIN_OP(vflt_w)
DEF_BIN_OP(vflt_d)

DEF_BIN_OP(vfle_w)
DEF_BIN_OP(vfle_d)
#endif

#define DEF_UNARY_OP_V(NAM) \
    template<vreg_no vd, vreg_no vs1, vop_type mode = vop_type::thread_all> inline void CONCAT_(NAM, _v)() \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).NAM(vd, vs1); \
    }

#if 0
DEF_UNARY_OP_V(vfsqrt)
#endif

#if 0
template<vreg_no vd, vreg_no vs1, vop_type mode = vop_type::thread_all> inline void vfclass_v();
#endif

#if 0
/// Floating-point reduction operations
///@{
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vfredosum_v();
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vfredsum_v();
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vfredmax_v();
template<vreg_no vd, vreg_no vs1, vop_type mode> inline void vfredmin_v();
///@}
#endif

/// Vector floating-point fused multiply-add
///@{
#define DEF_3_OP(NAM) \
    inline void NAM(vreg_no vd, vreg_no vs1, vreg_no vs2, vreg_no vs3, vop_type mode = vop_type::thread_all) \
    { \
        using namespace implementation; \
        static_cast<Operations&>(V_unit::instance().get_op_performer()).NAM(vd, vs1, vs2, vs3, mode); \
    }

#if 0
DEF_3_OP(vfmadd_vv)
DEF_3_OP(vfmadd_vf)

DEF_3_OP(vfmsub_vv)
DEF_3_OP(vfmsub_vf)
#endif

#undef DEF_3_OP
///@}

#if 0
/// Convert integer to narrower integer
///@{
//template<typename to, typename from, vop_type mode = vop_type::thread_all> void vcvt(vreg_no vd, vreg_no vs1);
///@}

/// Convert integer to float
///@{
template<typename to, typename from, vop_type mode = vop_type::thread_all> void vfcvt(vreg_no vd, vreg_no vs1);
///@}

/// Move to/from floating-point (f) registers.
///@{
template<typename to, typename from> void vfmv(vreg_no vd, from fs1);
template<typename to> to vfmv(vreg_no vs1);
///@}
#endif

#if 0
template<vreg_no vd, vop_type mode = vop_type::thread_all> inline void vmiota()
{
    using namespace implementation;
    static_cast<Operations<mode>&>(V_unit::instance()).vmiota(vd);
}
#endif

#if 0
template<vreg_no vd, vop_type mode = vop_type::thread_all> inline void vinsx(int32_t value, size_t idx = 0)
{
    using namespace implementation;
    static_cast<Operations<mode>&>(V_unit::instance()).vinsx(vd, value, idx);
}
#endif

#undef DEF_UNARY_OP
#undef DEF_BIN_IMM_OP
#undef DEF_BIN_OP_VV
#undef DEF_BIN_OP_VX
#undef DEF_BIN_OP_VI
#undef DEF_BIN_OP_VXI
#undef DEF_BIN_OP_VF

}  // namespace spec_0_7
}  // namespace v
}  // namespace riscv

#undef PASTE_
#undef CONCAT_

#endif  // RISCV_EXT_V_HPP_
