cmake_minimum_required(VERSION 3.6)

project(rvv-simulator)

set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.66)

enable_testing()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC)
    add_definitions(
        -D_SCL_SECURE_NO_WARNINGS
        -D_USE_MATH_DEFINES
        -DNOMINMAX
        )
    add_compile_options(/bigobj)
endif(MSVC)

set(RVV_ELEN 64 CACHE STRING "The maximum size of a single vector element in bits")
set(RVV_VLEN 256 CACHE STRING "The number of bits in a vector register") # VLEN ≥ ELEN
set(RVV_SLEN 64 CACHE STRING "The striping distance in bits") # VLEN ≥ SLEN ≥ 32

set(CXX_FEATURES
    #cxx_aggregate_default_initializers
    cxx_alias_templates
    cxx_alignas
    cxx_alignof
    cxx_attributes
    cxx_attribute_deprecated
    cxx_auto_type
    cxx_binary_literals
    cxx_constexpr
    cxx_contextual_conversions
    #cxx_decltype_incomplete_return_types
    cxx_decltype
    cxx_decltype_auto
    cxx_default_function_template_args
    cxx_defaulted_functions
    cxx_defaulted_move_initializers
    cxx_delegating_constructors
    cxx_deleted_functions
    cxx_digit_separators
    cxx_enum_forward_declarations
    cxx_explicit_conversions
    cxx_extended_friend_declarations
    cxx_extern_templates
    cxx_final
    cxx_func_identifier
    cxx_generalized_initializers
    cxx_generic_lambdas
    cxx_inheriting_constructors
    cxx_inline_namespaces
    cxx_lambdas
    cxx_lambda_init_captures
    cxx_local_type_template_args
    cxx_long_long_type
    cxx_noexcept
    cxx_nonstatic_member_init
    cxx_nullptr
    cxx_override
    cxx_range_for
    cxx_raw_string_literals
    cxx_reference_qualified_functions
    #cxx_relaxed_constexpr
    cxx_return_type_deduction
    cxx_right_angle_brackets
    cxx_rvalue_references
    cxx_sizeof_member
    cxx_static_assert
    cxx_strong_enums
    cxx_thread_local
    cxx_trailing_return_types
    cxx_unicode_literals
    cxx_uniform_initialization
    cxx_unrestricted_unions
    cxx_user_literals
    cxx_variable_templates
    cxx_variadic_macros
    cxx_variadic_templates
    cxx_template_template_parameters
)

add_library(riscv32 STATIC
    src/riscv32/v.cpp
    include/riscv/ext/v.hpp
)
target_compile_definitions(riscv32 PUBLIC -DRVV_ELEN=${RVV_ELEN} PUBLIC -DRVV_VLEN=${RVV_VLEN} PUBLIC -DRVV_SLEN=${RVV_SLEN})
set_target_properties(riscv32 PROPERTIES FOLDER "Libs")
target_include_directories(riscv32 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(riscv32 Boost::boost)
target_compile_features(riscv32 PUBLIC ${CXX_FEATURES})

add_subdirectory(vstdlib)
add_subdirectory(tests)
