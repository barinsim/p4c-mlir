#include "P4Ops.h"
#include "P4Dialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "P4Ops.cpp.inc"


mlir::LogicalResult p4mlir::ConstantOp::verify() {
    if (getValueAttr().getType() == getResult().getType()) {
        return mlir::success();
    }
    return mlir::emitError(
        getLoc(),
        "'p4.constant' op "
        "attribute 'value' failed to satisfy constraint: Attribute and result value "
        "must have a same type");
}
