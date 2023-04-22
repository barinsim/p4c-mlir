#include "P4Ops.h"
#include "P4Dialect.h"

#include <iostream>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/DialectImplementation.h"

#include "P4OpsEnumAttr.cpp.inc"
#include "P4OpsAttr.cpp.inc"
#define GET_OP_CLASSES
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

mlir::LogicalResult CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the callee symbol is in scope
    auto callee = getCalleeAttr();
    // TODO: Add more callable ops
    ActionOp act = symbolTable.lookupNearestSymbolFrom<ActionOp>(*this, callee);
    if (!act) {
        return emitOpError() << "'" << callee.getNestedReferences()
                             << "' does not reference a valid callable";
    }

    // Verify that the operand types match the callee
    mlir::TypeRange args = getOperands().getTypes();
    mlir::TypeRange params = act.getBody().getArgumentTypes();
    if (args.size() != params.size()) {
        return emitOpError("incorrect number of operands for callee");
    }
    if (args != params) {
        return emitOpError("argument types do not match the parameter types");
    }

    // Verify that the result types match the callee
    if (!getResults().empty()) {
        return emitOpError("Not implemented");
    }

    return mlir::success();
}

} // namespace p4mlir