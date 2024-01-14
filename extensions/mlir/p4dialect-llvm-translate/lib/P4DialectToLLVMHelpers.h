#ifndef P4C_P4DIALECTTOLLVMHELPERS_H
#define P4C_P4DIALECTTOLLVMHELPERS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Types.h"

mlir::Type createSignlessType(OpBuilder &builder, mlir::IntegerType type) {
    if (!type) {
        llvm::errs() << "expected valid type";
        return mlir::Type();
    }
    unsigned width = type.getWidth();
    return mlir::IntegerType::get(builder.getContext(), width, mlir::IntegerType::Signless);
}

mlir::Attribute createSignlessAttr(OpBuilder &builder, mlir::IntegerAttr attr) {
    auto type = attr.getType().dyn_cast<IntegerType>();
    if (!type) {
        llvm::errs() << "expected valid type";
        return mlir::Attribute();
    }
    auto signlessType = createSignlessType(builder, type);
    return builder.getIntegerAttr(signlessType, attr.getValue());
}

#endif  // P4C_P4DIALECTTOLLVMHELPERS_H
