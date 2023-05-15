set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_mlir_translation_library(p4c-mlir-translate-lib
  cfg.cpp
  domTree.cpp
  ssa.cpp
  mlirgen.cpp
  utils.cpp
  
  DEPENDS
  genIR

  LINK_LIBS
  ${P4C_LIB_DEPS}
  ${P4C_LIBRARIES}
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
)
target_include_directories(p4c-mlir-translate-lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

llvm_update_compile_flags(p4c-mlir-translate-lib)
mlir_check_link_libraries(p4c-mlir-translate-lib)