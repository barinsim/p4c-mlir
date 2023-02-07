set(P4C_MLIR_SOURCES main.cpp)

add_executable(p4c-mlir ${P4C_MLIR_SOURCES})
target_link_libraries (p4c-mlir ${P4C_LIBRARIES} ${P4C_LIB_DEPS})
add_dependencies(p4c-mlir genIR)