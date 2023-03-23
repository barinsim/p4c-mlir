# This is just a debug executable
add_llvm_executable(p4c-mlir main.cpp)
target_link_libraries (p4c-mlir p4c-mlir-translate-lib)