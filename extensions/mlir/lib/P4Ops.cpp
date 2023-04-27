#include "P4Ops.h"
#include "P4Dialect.h"

#include <iostream>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

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

void ActionOp::print(mlir::OpAsmPrinter &printer) {
    mlir::function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                   getResAttrsAttrName());
}

void ExternOp::print(mlir::OpAsmPrinter &printer) {
    mlir::function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                   getResAttrsAttrName());
}

mlir::ParseResult ActionOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

mlir::ParseResult ExternOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

namespace {

template <typename OpType>
void buildFuncLikeOp(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
    state.addAttribute(OpType::getFunctionTypeAttrName(state.name), mlir::TypeAttr::get(type));
    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty()) {
        return;
    }
    assert(type.getNumInputs() == argAttrs.size());
    mlir::function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs, std::nullopt,
                                                        OpType::getArgAttrsAttrName(state.name),
                                                        OpType::getResAttrsAttrName(state.name));
}

}

void ActionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<ActionOp>(builder, state, name, type, attrs, argAttrs);
}

void ExternOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<ExternOp>(builder, state, name, type, attrs, argAttrs);
}

mlir::ParseResult ControlOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

mlir::LogicalResult CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the callee symbol is in scope
    auto callee = getCalleeAttr();
    auto func = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(*this, callee);
    if (!func) {
        return emitOpError() << "'" << callee.getNestedReferences()
                             << "' does not reference a valid callable";
    }

    // Verify that the operand types match the callee
    mlir::TypeRange args = getOperands().getTypes();
    mlir::TypeRange params = func.getArgumentTypes();
    if (args.size() != params.size()) {
        return emitOpError("incorrect number of operands for callee");
    }
    if (args != params) {
        return emitOpError("argument types do not match the parameter types");
    }

    // Verify that the result types match the callee
    mlir::TypeRange callResults = getResults().getTypes();
    mlir::TypeRange funcResults = func.getResultTypes();
    if (callResults.size() != funcResults.size()) {
        return emitOpError("incorrect number of results for callee");
    }
    if (callResults != funcResults) {
        return emitOpError("call return types do not match the declared return types");
    }

    return mlir::success();
}

} // namespace p4mlir