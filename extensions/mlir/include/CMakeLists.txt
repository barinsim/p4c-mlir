# `add_mlir_dialect` does not generate attributes and enums,
# they must be added right before `add_mlir_dialect` to be
# added to the correct target
set(LLVM_TARGET_DEFINITIONS P4Ops.td)
mlir_tablegen(P4OpsEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(P4OpsEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(P4OpsAttr.h.inc -gen-attrdef-decls)
mlir_tablegen(P4OpsAttr.cpp.inc -gen-attrdef-defs)

# Adds mlir dialect which is defined in P4Ops.td.
# Sets up a target which calls TableGen.
# This target is a dependency of MLIRP4Dialect,
# which finally gets linked into tools like -opt or -translate.
add_mlir_dialect(P4Ops p4)

add_mlir_doc(P4Dialect P4Dialect P4/ -gen-dialect-doc)
add_mlir_doc(P4Ops P4Ops P4/ -gen-op-doc)