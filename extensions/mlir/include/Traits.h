#ifndef P4C_TRAITS_H
#define P4C_TRAITS_H

#include <list>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpDefinition.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"

template <typename ConcreteType>
class Template : public mlir::OpTrait::TraitBase<ConcreteType, Template> {
 public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
        if (op->getNumRegions() != 1) {
            return op->emitOpError()
                   << "Operations with a 'Template' must have exactly one region";
        }
        if (!op->getRegion(0).hasOneBlock() && !op->getRegion(0).empty()) {
            return op->emitOpError()
                   << "Operations with a 'Template' must have one or zero block";
        }
        if (!op->hasAttrOfType<mlir::ArrayAttr>("type_parameters")) {
            return op->emitOpError()
                   << "Operations with a 'Template' must have 'type_parameters' argument "
                   << "of type 'ArrayRef<mlir::StringAttr>'";
        }
        // TODO: checks the params are unique

        // Collect types used within this op (regions are not included)
        // TODO: just check 'function_type'
        std::vector<mlir::Type> types;
        auto resTypes = op->getResultTypes();
        std::copy(resTypes.begin(), resTypes.end(), std::back_inserter(types));
        auto argTypes = op->getOperandTypes();
        std::copy(argTypes.begin(), argTypes.end(), std::back_inserter(types));
        auto allAttrs = op->getAttrs();
        std::for_each(allAttrs.begin(), allAttrs.end(), [&](mlir::NamedAttribute namedAttr) {
            auto attr = namedAttr.getValue();
            if (auto typeAttr = attr.dyn_cast_or_null<mlir::TypeAttr>()) {
                types.push_back(typeAttr.getValue());
            }
            if (auto arrayAttr = attr.dyn_cast_or_null<mlir::ArrayAttr>()) {
                std::for_each(arrayAttr.begin(), arrayAttr.end(), [&](mlir::Attribute attr) {
                    if (auto typeAttr = attr.dyn_cast_or_null<mlir::TypeAttr>()) {
                        types.push_back(typeAttr.getValue());
                    }
                });
            }

        });

        // Extract inner types and filter type variables
        std::list<p4mlir::TypeVarType> typeVars;
        std::for_each(types.begin(), types.end(), [&](mlir::Type type) {
            // TODO: extern_class types, refactor with getInnerTypes()
            auto extract = [&](mlir::Type type) {
                if (auto refType = type.dyn_cast_or_null<p4mlir::RefType>()) {
                    type = refType.getType();
                }
                if (auto typeVarType = type.dyn_cast_or_null<p4mlir::TypeVarType>()) {
                    typeVars.push_back(typeVarType);
                }
            };

            if (auto funcType = type.dyn_cast_or_null<mlir::FunctionType>()) {
                auto results = funcType.getResults();
                std::for_each(results.begin(), results.end(), extract);
                auto inputs = funcType.getInputs();
                std::for_each(inputs.begin(), inputs.end(), extract);
            } else {
                extract(type);
            }
        });

        // Walk chain of 'Template' ops upwards and remove resolved type variables from the worklist
        Operation* ptr = op;
        while (ptr && ptr->hasTrait<Template>()) {
            auto typeSymbols = ptr->getAttrOfType<mlir::ArrayAttr>("type_parameters");
            if (!typeSymbols) {
                break;
            }
            for (auto it = typeVars.begin(); it != typeVars.end(); ) {
                auto attr = mlir::StringAttr::get(op->getContext(), it->getName());
                auto found = std::find(typeSymbols.begin(), typeSymbols.end(), attr);
                if (found != typeSymbols.end()) {
                    it = typeVars.erase(it);
                } else {
                    ++it;
                }
            }
            ptr = ptr->getParentOp();
        }

        if (!typeVars.empty()) {
            return op->emitOpError() << "Using undeclared type variable";
        }
        return mlir::success();
    }
};

#endif  // P4C_TRAITS_H
