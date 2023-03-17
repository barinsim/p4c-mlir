# Adds mlir dialect which is defined in P4Ops.td.
# Sets up a target which calls TableGen.
# This target is a dependency of MLIRP4Dialect,
# which finally gets linked into tools like -opt or -translate.
add_mlir_dialect(P4Ops p4)

add_mlir_doc(P4Dialect P4Dialect P4/ -gen-dialect-doc)
add_mlir_doc(P4Ops P4Ops P4/ -gen-op-doc)