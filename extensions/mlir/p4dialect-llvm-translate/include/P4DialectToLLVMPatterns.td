include "P4Ops.td"
include "mlir/Dialect/LLVMIR/LLVMOps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"
include "mlir/IR/BuiltinAttributes.td"

include "mlir/IR/PatternBase.td"

def SameType : Constraint<CPred<"$0.getType() == $1.getType()">, "has same type">;

// Propagate p4.copy
def : Pat<(CopyOp $value), (replaceWithValue $value)>;

// Eliminate no-op p4.cast
def : Pat<(CastOp:$res $value), (replaceWithValue $value), [(SameType $res, $value)]>;
