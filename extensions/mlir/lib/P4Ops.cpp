#include "P4Ops.h"
#include "P4Dialect.h"

#include <iostream>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/BuiltinOps.h"

#include "P4OpsEnumAttr.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "P4OpsAttr.cpp.inc"
#define GET_OP_CLASSES
#include "P4Ops.cpp.inc"


namespace p4mlir {

namespace {

// Convenience function to build ops with 'FunctionOpInterface' trait
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

// Find the closest parent ModuleOp, starting from 'from'
mlir::ModuleOp getParentModule(Operation* from) {
    auto moduleOp = from->getParentOfType<mlir::ModuleOp>();
    if (!moduleOp) {
        from->emitOpError("could not find parent mlir::ModuleOp");
    }
    return moduleOp;
}

} // namespace

mlir::LogicalResult ConstantOp::verify() {
    if (auto intAttr = dyn_cast<IntegerAttr>(getValueAttr())) {
        auto type = intAttr.getType();
        if (type != getResult().getType()) {
            return emitOpError("attribute and result must have the same type");
        }
        return mlir::success();
    }
    if (auto symbolAttr = dyn_cast<SymbolRefAttr>(getValueAttr())) {
        auto resType = getResult().getType();
        mlir::LogicalResult result =
            llvm::TypeSwitch<mlir::Type, mlir::LogicalResult>(resType)
                .Case([](p4mlir::ErrorType) { return mlir::success(); })
                .Case([](p4mlir::EnumType) { return mlir::success(); })
                .Default([&](mlir::Type) {
                    return emitOpError(
                        "symbol reference attributes must have result of type ErrorType");
                });
        return result;
    }
    return emitOpError("value attribute must be IntegerAttr or SymbolRefAttr");
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
    printer.printAttributeWithoutType(getValue());
    printer << " : ";
    printer.printType(getResult().getType());
}

mlir::ParseResult ControlPlaneValueOp::parse(mlir::OpAsmParser &parser,
                                             mlir::OperationState &result) {
    mlir::TypeAttr type;
    if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
        parser.parseAttribute(type, "type", result.attributes))
        return mlir::failure();

    result.addTypes(type.getValue());
    return mlir::success();
}

void ControlPlaneValueOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"type"});
    printer << ": ";
    printer << getType();
}

void ControlOp::print(mlir::OpAsmPrinter &printer) {
    auto funcName =
        getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);

    auto printArgs = [&](auto first, auto end) {
        printer << '(';
        while (first != end) {
            printer.printRegionArgument(*first);
            ++first;
            if (first != end) {
                printer.getStream() << ", ";
            }
        }
        printer << ')';
    };

    // Print apply arguments and optionally constructor arguments
    auto args = getBody().getArguments();
    std::size_t applyArgsCnt = getApplyType().getInputs().size();
    printArgs(args.begin(), args.begin() + applyArgsCnt);
    if (args.begin() + applyArgsCnt != args.end()) {
        printArgs(args.begin() + applyArgsCnt, args.end());
    }

    printer << ' ';
    printer.printRegion(getBody(), false, true);
}

void CallOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    printer.printAttributeWithoutType(getCalleeAttr());
    auto typeOperands = getTypeOperands();
    if (typeOperands) {
        printer << '<';
        printer.printType(typeOperands->begin()->cast<mlir::TypeAttr>().getValue());
        std::for_each(typeOperands->begin() + 1, typeOperands->end(), [&](mlir::Attribute attr) {
            printer << ", ";
            printer.printType(attr.cast<TypeAttr>().getValue());
        });
        printer << '>';
    }
    printer << "(";
    printer << getOperands();
    printer << ")";
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("callee");
    elidedAttrs.push_back("type_operands");
    printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    printer << " : ";
    printer.printFunctionalType(getOperands().getTypes(), getOperation()->getResultTypes());
}

mlir::ParseResult CallOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ExternOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    printer.printSymbolName(getSymName());
    auto typeParams = getTypeParameters();
    if (typeParams && !typeParams->empty()) {
        printer << '<';
        printer.printSymbolName(typeParams->begin()->dyn_cast_or_null<mlir::StringAttr>());
        std::for_each(typeParams->begin() + 1, typeParams->end(), [&](Attribute attr) {
            printer << ", ";
            printer.printSymbolName(attr.dyn_cast_or_null<mlir::StringAttr>());
        });
        printer << '>';
    }
    auto type = getFunctionType();
    function_interface_impl::printFunctionSignature(printer, *this, type.getInputs(), false,
                                                    type.getResults());
}

mlir::ParseResult ExternOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ExternClassOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    printer.printSymbolName(getSymName());
    auto typeParams = getTypeParameters();
    if (typeParams && !typeParams->empty()) {
        printer << '<';
        printer.printSymbolName(typeParams->begin()->dyn_cast_or_null<mlir::StringAttr>());
        std::for_each(typeParams->begin() + 1, typeParams->end(), [&](Attribute attr) {
            printer << ", ";
            printer.printSymbolName(attr.dyn_cast_or_null<mlir::StringAttr>());
        });
        printer << '>';
    }
    printer << ' ';
    printer.printRegion(getBody());
}

mlir::ParseResult ExternClassOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ActionOp::print(mlir::OpAsmPrinter &printer) {
    mlir::function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                   getResAttrsAttrName());
}

mlir::ParseResult ActionOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ActionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<ActionOp>(builder, state, name, type, attrs, argAttrs);
}

void ExternOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, ::mlir::ArrayAttr typeParams,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<ExternOp>(builder, state, name, type, attrs, argAttrs);
    state.addAttribute(getTypeParametersAttrName(state.name).getValue(), typeParams);
}

mlir::ParseResult ControlOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void TypeVarType::print(::mlir::AsmPrinter &printer) const {
    printer << '<';
    printer.printSymbolName(getName());
    printer << '>';
}

::mlir::Type TypeVarType::parse(::mlir::AsmParser &odsParser) {
    // TODO
    return mlir::Type();
}

mlir::LogicalResult CallOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the callee symbol is in scope.
    // Referenced 'callee' symbol is assumed
    // to be fully qualified within the closest parent module
    auto callee = getCalleeAttr();
    auto func = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
        getParentModule(*this), callee);
    if (!func) {
        return emitOpError() << "'" << callee.getNestedReferences()
                             << "' does not reference a valid callable";
    }

    // TODO: check template function calls
    std::vector<mlir::Type> types;
    std::copy(func.getArgumentTypes().begin(), func.getArgumentTypes().end(),
              std::back_inserter(types));
    std::copy(func.getResultTypes().begin(), func.getResultTypes().end(),
              std::back_inserter(types));
    if (std::any_of(types.begin(), types.end(), [](mlir::Type type) {
            if (type.isa<p4mlir::RefType>()) {
                type = type.cast<p4mlir::RefType>().getType();
            }
            return type.isa<p4mlir::TypeVarType>();
        })) {
        return mlir::success();
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
    mlir::TypeRange funcResults = func.getFunctionType().cast<mlir::FunctionType>().getResults();
    if (callResults.size() != funcResults.size()) {
        return emitOpError("incorrect number of results for callee");
    }
    if (callResults != funcResults) {
        return emitOpError("call return types do not match the declared return types");
    }

    return mlir::success();
}

void CallMethodOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    printer << getBase();
    printer << ' ';
    printer.printAttributeWithoutType(getCalleeAttr());
    auto typeOperands = getTypeOperands();
    if (typeOperands) {
        printer << '<';
        printer.printType(typeOperands->begin()->cast<mlir::TypeAttr>().getValue());
        std::for_each(typeOperands->begin() + 1, typeOperands->end(), [&](mlir::Attribute attr) {
            printer << ", ";
            printer.printType(attr.cast<TypeAttr>().getValue());
        });
        printer << '>';
    }
    printer << "(";
    printer << getOpers();
    printer << ")";
    ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
    elidedAttrs.push_back("callee");
    elidedAttrs.push_back("type_operands");
    printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
    printer << " : ";
    printer.printFunctionalType(getOperands().getTypes(), getOperation()->getResultTypes());
}

mlir::ParseResult CallMethodOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

mlir::LogicalResult CallMethodOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the callee symbol is in scope.
    // Referenced 'callee' symbol is assumed
    // to be fully qualified within the closest parent module
    auto callee = getCalleeAttr();
    auto func = symbolTable.lookupNearestSymbolFrom<mlir::FunctionOpInterface>(
        getParentModule(*this), callee);
    if (!func) {
        return emitOpError() << "'" << callee.getNestedReferences()
                             << "' does not reference a valid callable";
    }

    // TODO: check template method calls
    std::vector<mlir::Type> types;
    std::copy(func.getArgumentTypes().begin(), func.getArgumentTypes().end(),
              std::back_inserter(types));
    std::copy(func.getResultTypes().begin(), func.getResultTypes().end(),
              std::back_inserter(types));
    if (std::any_of(types.begin(), types.end(), [](mlir::Type type) {
            if (type.isa<p4mlir::RefType>()) {
                type = type.cast<p4mlir::RefType>().getType();
            }
            return type.isa<p4mlir::TypeVarType>();
        })) {
        return mlir::success();
    }

    // Verify that the operand types match the callee
    mlir::TypeRange args = getOpers().getTypes();
    mlir::TypeRange params = func.getArgumentTypes();
    if (args.size() != params.size()) {
        return emitOpError("incorrect number of operands for callee");
    }
    if (args != params) {
        return emitOpError("argument types do not match the parameter types");
    }

    // Verify that the result types match the callee
    mlir::TypeRange callResults = getResults().getTypes();
    mlir::TypeRange funcResults = func.getFunctionType().cast<mlir::FunctionType>().getResults();
    if (callResults.size() != funcResults.size()) {
        return emitOpError("incorrect number of results for callee");
    }
    if (callResults != funcResults) {
        return emitOpError("call return types do not match the declared return types");
    }

    return mlir::success();
}

void ConstructorOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    printer.printSymbolName(getSymName());
    auto type = getFunctionType();
    function_interface_impl::printFunctionSignature(printer, *this, type.getInputs(), false,
                                                    type.getResults());
}

mlir::ParseResult ConstructorOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void ConstructorOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<ConstructorOp>(builder, state, name, type, attrs, argAttrs);
}

void TableOp::print(mlir::OpAsmPrinter &printer) {
    auto tableName = getSymNameAttr().getValue();
    printer << ' ';
    printer.printSymbolName(tableName);

    auto printArgs = [&](auto first, auto end) {
        printer << '(';
        while (first != end) {
            printer.printRegionArgument(*first);
            ++first;
            if (first != end) {
                printer.getStream() << ", ";
            }
        }
        printer << ')';
    };

    // Print apply arguments
    auto args = getBody().getArguments();
    printArgs(args.begin(), args.end());

    printer << ' ';
    printer.printRegion(getBody(), false, true);
}

mlir::ParseResult TableOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void TablePropertyOp::print(mlir::OpAsmPrinter &printer) {
    auto tableName = getSymNameAttr().getValue();
    printer << ' ';
    printer.printSymbolName(tableName);
    printer << ' ';
    printer.printRegion(getBody(), false, true);
}

mlir::ParseResult TablePropertyOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

mlir::ParseResult TableActionOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void TableActionOp::print(mlir::OpAsmPrinter &printer) {
    printer << ' ';
    auto actionName = getActionNameAttr();
    if (actionName) {
        printer.printAttributeWithoutType(actionName);
    }
    if (!getBody().empty()) {
        printer.printRegion(getBody(), false, true);
    }
}

mlir::LogicalResult TableKeyOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the name of the match kind is in scope
    auto matchKind = getMatchKind();
    auto matchKindDecl = symbolTable.lookupNearestSymbolFrom<p4mlir::MatchKindOp>(
        getParentModule(*this), matchKind);
    if (!matchKind) {
        return emitOpError() << "'" << matchKind
                             << "' does not reference a valid match kind";
    }

    return mlir::success();
}

mlir::ParseResult DontcareOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    mlir::TypeAttr type;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(type, "type", result.attributes))
        return mlir::failure();

    result.addTypes(type.getValue());
    return mlir::success();
}

void DontcareOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), {"type"});
    printer << ": ";
    printer << getType();
}

void ParserOp::print(mlir::OpAsmPrinter &printer) {
    auto funcName =
        getSymNameAttr().getValue();

    printer << ' ';
    printer.printSymbolName(funcName);

    auto printArgs = [&](auto first, auto end) {
        printer << '(';
        while (first != end) {
            printer.printRegionArgument(*first);
            ++first;
            if (first != end) {
                printer.getStream() << ", ";
            }
        }
        printer << ')';
    };

    // Print apply arguments and optionally constructor arguments
    auto args = getBody().getArguments();
    std::size_t applyArgsCnt = getApplyType().getInputs().size();
    printArgs(args.begin(), args.begin() + applyArgsCnt);
    if (args.begin() + applyArgsCnt != args.end()) {
        printArgs(args.begin() + applyArgsCnt, args.end());
    }

    printer << ' ';
    printer.printRegion(getBody(), false, true);
}

mlir::ParseResult ParserOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void StateOp::print(mlir::OpAsmPrinter &printer) {
    mlir::function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false,
                                                   getFunctionTypeAttrName(), getArgAttrsAttrName(),
                                                   getResAttrsAttrName());
}

mlir::ParseResult StateOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
    // TODO:
    return mlir::failure();
}

void StateOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, llvm::StringRef name,
                     mlir::FunctionType type, llvm::ArrayRef<mlir::NamedAttribute> attrs,
                     llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
    buildFuncLikeOp<StateOp>(builder, state, name, type, attrs, argAttrs);
}

mlir::LogicalResult TransitionOp::verifySymbolUses(::mlir::SymbolTableCollection& symbolTable) {
    // Verify that the state symbol is in scope.
    // Referenced 'state' symbol is assumed
    // to be fully qualified within the closest parent module
    auto stateName = getStateAttr();
    auto state = symbolTable.lookupNearestSymbolFrom<p4mlir::StateOp>(
        getParentModule(*this), stateName);
    if (!state) {
        return emitOpError() << "'" << stateName << "' does not reference a valid state";
    }

    std::vector<mlir::Type> types;
    std::copy(state.getArgumentTypes().begin(), state.getArgumentTypes().end(),
              std::back_inserter(types));
    std::copy(state.getResultTypes().begin(), state.getResultTypes().end(),
              std::back_inserter(types));

    // Verify that the operand types match the state declaration
    mlir::TypeRange args = getOperands().getTypes();
    mlir::TypeRange params = state.getArgumentTypes();
    if (args.size() != params.size()) {
        return emitOpError("incorrect number of operands for state");
    }
    if (args != params) {
        return emitOpError("argument types do not match the parameter types");
    }

    return mlir::success();
}

} // namespace p4mlir
