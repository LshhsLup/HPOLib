add_executable(test_half tensor/test_half.cu)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORTS_F16C)

# half-precision floating point(half.h) requires F16C support on x86 architectures
if(COMPILER_SUPPORTS_F16C)
    target_compile_options(test_half PRIVATE -mf16c)
elseif(MSVC)
    target_compile_options(test_half PRIVATE /arch:AVX2)
endif()

target_link_libraries(test_half PRIVATE
    hpolib      
    gtest_main     
)

include(GoogleTest)
gtest_discover_tests(test_half)