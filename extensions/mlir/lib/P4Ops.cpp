#include "P4Ops.h"
#include "P4Dialect.h"

#include <iostream>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_CLASSES
#include "P4OpsEnumAttr.cpp.inc"
#include "P4OpsAttr.cpp.inc"
#include "P4Ops.cpp.inc"


namespace p4mlir {


mlir::LogicalResult ConstantOp::verify() {
    if (getValueAttr().getType() == getResult().getType()) {
        return mlir::success();
    }
    return mlir::emitError(
        getLoc(),
        "'p4.constant' op "
        "attribute 'value' failed to satisfy constraint: Attribute and result value "
        "must have a same type");
}

mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    mlir::IntegerAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return mlir::failure();

    result.addTypes(value.getType());
    return mlir::success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"value"});
    printer << getValue();
}

void ActionOp::print(mlir::OpAsmPrinter &printer) {
    auto funcName =
        getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);

    printer << '(';
    auto args = getBody().getArguments();
    if (!args.empty()) {
        std::for_each(args.begin(), args.end() - 1, [&](auto arg) {
            printer.printRegionArgument(arg);
            printer.getStream() << ", ";
        });
        printer.printRegionArgument(args.back());
    }

    printer << ')';
    printer << ' ';

    printer.printRegion(getBody(), false, true);
}

mlir::ParseResult ActionOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ControlOp::print(mlir::OpAsmPrinter &printer) {
    auto funcName =
        getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);
    printer << '(';
    auto args = getBody().getArguments();
    if (!args.empty()) {
        std::for_each(args.begin(), args.end() - 1, [&](auto arg) {
            printer.printRegionArgument(arg);
            printer.getStream() << ", ";
        });
        printer.printRegionArgument(args.back());
    }
    printer << ')';
    printer << ' ';

    printer.printRegion(getBody(), false, true);
}

mlir::ParseResult ControlOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

} // namespace p4mlir