add_subdirectory(lib)
add_subdirectory(debug)

add_llvm_executable(p4c-mlir-translate p4c-mlir-translate.cpp)
llvm_update_compile_flags(p4c-mlir-translate)
target_link_libraries(p4c-mlir-translate p4c-mlir-translate-lib)

target_link_libraries(p4c-mlir-translate MLIRP4DialectToLLVM)

mlir_check_link_libraries(p4c-mlir-translate)


