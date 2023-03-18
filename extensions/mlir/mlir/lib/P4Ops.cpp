#include "P4Ops.h"
#include "P4Dialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "P4Ops.cpp.inc"


mlir::LogicalResult p4mlir::ConstantOp::verify() {
    return mlir::success();
}

void p4mlir::ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                               int value) {
    auto type = mlir::IntegerType::get(odsBuilder.getContext(), 32, mlir::IntegerType::Signless);
    build(odsBuilder, odsState, mlir::IntegerAttr::get(type, value));
}