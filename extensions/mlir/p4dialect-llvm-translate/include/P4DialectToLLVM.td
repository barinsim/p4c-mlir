include "P4Ops.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"

include "mlir/IR/PatternBase.td"

def : Pat<(AddOp $lhs, $rhs), (LLVM_AddOp $lhs, $rhs)>;