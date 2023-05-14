#include "mlirgen.h"

#include "frontends/p4/methodInstance.h"


namespace p4mlir {

namespace {

// Converts P4 location stored in 'loc' into its MLIR counterpart
mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node) {
    // TODO:
    CHECK_NULL(node);
    return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
}

// Turns MLIR type T into MLIR type `!p4.ref<T>`
mlir::Type wrappedIntoRef(mlir::OpBuilder& builder, mlir::Type type) {
    BUG_CHECK(!type.isa<RefType>(), "Ref type cannot be wrapped into another reference");
    return p4mlir::RefType::get(builder.getContext(), type);
}

// Converts P4 type into corresponding MLIR type
mlir::Type toMLIRType(mlir::OpBuilder& builder, const IR::Type* p4type) {
    CHECK_NULL(p4type);
    if (p4type->is<IR::Type_InfInt>()) {
        // TODO: create special type
        return mlir::IntegerType::get(builder.getContext(), 64, mlir::IntegerType::Signed);
    }
    else if (auto* bits = p4type->to<IR::Type_Bits>()) {
        auto sign = bits->isSigned ? mlir::IntegerType::Signed : mlir::IntegerType::Unsigned;
        int size = bits->size;
        return mlir::IntegerType::get(builder.getContext(), size, sign);
    }
    else if (p4type->is<IR::Type_Boolean>()) {
        return mlir::IntegerType::get(builder.getContext(), 1, mlir::IntegerType::Signless);
    }
    else if (auto* hdr = p4type->to<IR::Type_Header>()) {
        cstring name = hdr->name;
        auto type = p4mlir::HeaderType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        BUG_CHECK(type, "Could not retrieve Header type");
        return type;
    }
    else if (auto* str = p4type->to<IR::Type_Struct>()) {
        cstring name = str->name;
        auto type = p4mlir::StructType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        BUG_CHECK(type, "Could not retrieve Struct type");
        return type;
    }
    else if (auto* control = p4type->to<IR::Type_Control>()) {
        cstring name = control->name;
        auto type = p4mlir::ControlType::get(builder.getContext(), llvm::StringRef(name.c_str()));
        BUG_CHECK(type, "Could not retrieve Control type");
        return type;
    }
    else if (auto* method = p4type->to<IR::Type_MethodBase>()) {
        std::vector<mlir::Type> paramTypes;
        std::vector<mlir::Type> retTypes;
        auto* params = method->parameters;
        std::transform(params->begin(), params->end(), std::back_inserter(paramTypes),
                       [&](auto *param) {
                           auto type = toMLIRType(builder, param->type);
                           auto dir = param->direction;
                           if (dir == IR::Direction::Out || dir == IR::Direction::InOut) {
                               type = wrappedIntoRef(builder, type);
                           }
                           return type;
                       });
        if (method->returnType && !method->returnType->is<IR::Type_Void>()) {
            retTypes.push_back(toMLIRType(builder, method->returnType));
        }
        return builder.getFunctionType(paramTypes, retTypes);
    }

    throw std::domain_error("Not implemented");
    return nullptr;
}

// Creates block arguments for jump from 'bb' to 'succ'.
// The order of arguments corresponds to phi arguments stored within 'ssaInfo'.
// 'ssaInfo' should be also used to create block parameters to match the order
std::vector<mlir::Value> createBlockArgs(const SSAInfo &ssaInfo, const BasicBlock *bb,
                                         const BasicBlock *succ,
                                         const ValuesTable& valuesTable) {
    CHECK_NULL(bb, succ);
    std::vector<mlir::Value> rv;
    auto phiInfo = ssaInfo.getPhiInfo(succ);
    for (auto &[decl, phi] : phiInfo) {
        BUG_CHECK(phi.sources.count(bb), "Phi node does not contain argument for the block");
        auto id = phi.sources.at(bb).value();
        auto argVal = valuesTable.get(decl, id);
        rv.push_back(argVal);
    }
    return rv;
}

// Return true if 'value' is of type 'RefType'
bool isRef(mlir::Value value) {
    return value.getType().isa<p4mlir::RefType>();
}

} // namespace

void MLIRGenImplCFG::postorder(const IR::BoolLiteral* boolean) {
    auto type = toMLIRType(builder, typeMap->getType(boolean));
    CHECK_NULL(type);
    mlir::Value val = builder.create<p4mlir::ConstantOp>(loc(builder, boolean), type,
                                                          (int64_t)boolean->value);
    valuesTable.add(boolean, val);
}

void MLIRGenImplCFG::postorder(const IR::Constant* cst) {
    auto type = toMLIRType(builder, typeMap->getType(cst));
    CHECK_NULL(type);
    BUG_CHECK(cst->fitsInt64(), "Not implemented");
    mlir::Value val =
        builder.create<p4mlir::ConstantOp>(loc(builder, cst), type, cst->asInt64());
    valuesTable.add(cst, val);
}

void MLIRGenImplCFG::postorder(const IR::ReturnStatement* ret) {
    builder.create<p4mlir::ReturnOp>(loc(builder, ret));
}

void MLIRGenImplCFG::postorder(const IR::AssignmentStatement* assign) {
    auto rValue = valuesTable.get(assign->right);

    const IR::IDeclaration* decl = nullptr;
    if (auto* pe = assign->left->to<IR::PathExpression>()) {
        CHECK_NULL(pe->path);
        decl = refMap->getDeclaration(pe->path, true);
    }

    // Write to an SSA register allocated variable
    if (decl && allocation.get(decl) == AllocType::REG) {
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
        ID ssaID = ssaInfo.getID(assign->left->to<IR::PathExpression>());
        valuesTable.add(decl, ssaID, value);
        return;
    }

    // Write to a stack allocated variable
    mlir::Value addr = valuesTable.getAddr(assign->left);
    builder.create<p4mlir::StoreOp>(loc(builder, assign), addr, rValue);
}

void MLIRGenImplCFG::postorder(const IR::Declaration_Variable* decl) {
    // Retrieves or creates init value
    auto createInitValue = [&](auto* decl) -> mlir::Value {
        if (!decl->initializer) {
            auto type = toMLIRType(builder, typeMap->getType(decl));
            auto init = builder.create<p4mlir::UninitializedOp>(loc(builder, decl), type);
            return init;
        }
        return valuesTable.get(decl->initializer);
    };

    auto init = createInitValue(decl);

    // No need to allocate stack space for reg variables
    if (allocation.get(decl) == AllocType::REG) {
        auto value = init;
        if (decl->initializer) {
            auto type = toMLIRType(builder, typeMap->getType(decl));
            value = builder.create<p4mlir::CopyOp>(loc(builder, decl), type, init);
        }
        ID ssaID = ssaInfo.getID(decl);
        valuesTable.add(decl, ssaID, value);
        return;
    }

    // Create space for stack allocated variables
    BUG_CHECK(allocation.get(decl) == AllocType::STACK, "Expected STACK allocation");
    auto type = toMLIRType(builder, typeMap->getType(decl));
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr = builder.create<p4mlir::AllocOp>(loc(builder, decl), refType);
    builder.create<p4mlir::StoreOp>(loc(builder, decl), addr, init);
    valuesTable.addAddr(decl, addr);
}

void MLIRGenImplCFG::postorder(const IR::Cast* cast) {
    CHECK_NULL(cast->destType);
    auto src = valuesTable.get(cast->expr);
    auto targetType = toMLIRType(builder, cast->destType);
    mlir::Value value = builder.create<p4mlir::CastOp>(loc(builder, cast), targetType, src);
    valuesTable.add(cast, value);
}

void MLIRGenImplCFG::postorder(const IR::Operation_Relation* cmp) {
    auto lhsValue = valuesTable.get(cmp->left);
    auto rhsValue = valuesTable.get(cmp->right);

    auto selectKind = [](const IR::Operation_Relation* cmp) {
        if (cmp->is<IR::Equ>()) {
            return CompareOpKind::eq;
        } else if (cmp->is<IR::Neq>()) {
            return CompareOpKind::ne;
        } else if (cmp->is<IR::Lss>()) {
            return CompareOpKind::lt;
        } else if (cmp->is<IR::Leq>()) {
            return CompareOpKind::le;
        } else if (cmp->is<IR::Grt>()) {
            return CompareOpKind::gt;
        } else if (cmp->is<IR::Geq>()) {
            return CompareOpKind::ge;
        }
        BUG_CHECK(false, "Unknown comparison operator");
    };

    CompareOpKind kind = selectKind(cmp);

    auto res =
        builder.create<CompareOp>(loc(builder, cmp), kind, lhsValue, rhsValue);
    valuesTable.add(cmp, res);
}

void MLIRGenImplCFG::postorder(const IR::Member* mem) {
    // Callable reference does not generate any ops, skip it
    auto* p4type = typeMap->getType(mem, true);
    if (p4type->is<IR::Type_Method>()) {
        return;
    }

    mlir::Value baseValue = valuesTable.getUnchecked(mem->expr);
    auto type = toMLIRType(builder, p4type);
    auto name = builder.getStringAttr(mem->member.toString().c_str());

    // Materialize member of a register allocated variable
    if (!isRef(baseValue)) {
        BUG_CHECK(isRead(),
                  "Member access to a register variable can be used only in a read context");
        mlir::Value val =
            builder.create<p4mlir::GetMemberOp>(loc(builder, mem), type, baseValue, name);
        valuesTable.add(mem, val);
        return;
    }

    BUG_CHECK(isRef(baseValue), "Expected STACK allocated base object");

    // Retrieve member reference of a stack allocated variable
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr =
        builder.create<p4mlir::GetMemberRefOp>(loc(builder, mem), refType, baseValue, name);

    // Member variable is written, return just the reference
    if (isWrite()) {
        valuesTable.addAddr(mem, addr);
        return;
    }

    // Member variable is only read.
    // If this is the last member access within the path, materialize the value.
    // Otherwise return just a reference
    if (findContext<IR::Member>()) {
        valuesTable.addAddr(mem, addr);
        return;
    }
    mlir::Value val = builder.create<p4mlir::LoadOp>(loc(builder, mem), type, addr);
    valuesTable.add(mem, val);
}

void MLIRGenImplCFG::postorder(const IR::MethodCallExpression* call) {
    // 'MethodCallExpression' represents different types of calls, each of which
    // needs to generate different ops.
    // Figure out which call this is and generate correct mlir
    auto* instance = P4::MethodInstance::resolve(call, refMap, typeMap);
    auto loca = loc(builder, call);
    CHECK_NULL(instance);

    // Resolve call arguments to MLIR values
    std::vector<mlir::Value> operands;
    auto* args = call->arguments;
    auto* parameters = instance->originalMethodType->parameters;
    BUG_CHECK(args->size() == parameters->size(), "Args and params differ in size");
    int size = args->size();
    for (int i = 0; i < size; ++i) {
        auto* param = parameters->getParameter(i);
        auto* arg = args->at(i);
        auto dir = param->direction;
        if (dir == IR::Direction::None || dir == IR::Direction::In) {
            operands.push_back(valuesTable.get(arg));
        } else {
            operands.push_back(valuesTable.getAddr(arg));
        }
    }

    // Generate depending on type of the call
    if (auto* actCall = instance->to<P4::ActionCall>()) {
        auto name = symbols.getSymbol(actCall->action);
        builder.create<p4mlir::CallOp>(loca, name, operands);
    }
    else if (auto* builtin = instance->to<P4::BuiltInMethod>()) {
        BUG_CHECK(typeMap->getType(builtin->appliedTo, true)->is<IR::Type_Header>(),
                  "Not implemented");
        auto loca = loc(builder, call);
        cstring name = builtin->name;
        auto type = builder.getIntegerType(1);

        // Generates ops to retrieve the `__valid` member, either as a p4.ref<> or loaded value.
        // Returns the retrieved value
        auto getValidBit = [&](mlir::Value base) -> mlir::Value {
            auto fieldName = builder.getStringAttr("__valid");
            if (isRef(base)) {
                auto refType = wrappedIntoRef(builder, type);
                return builder.create<p4mlir::GetMemberRefOp>(loca, refType, base, fieldName);
            }
            return builder.create<p4mlir::GetMemberOp>(loca, type, base, fieldName);
        };

        auto member = getValidBit(valuesTable.getUnchecked(builtin->appliedTo));

        // Translate builtins of the header type into explicit operations on the __valid field
        if (name == "setValid" || name == "setInvalid") {
            auto cst = builder.create<p4mlir::ConstantOp>(loca, type, name == "setValid");
            builder.create<p4mlir::StoreOp>(loca, member, cst);
        } else if (name == "isValid") {
            mlir::Value value = member;
            if (isRef(value)) {
                value = builder.create<p4mlir::LoadOp>(loca, type, member);
            }
            valuesTable.add(call, value);
        } else {
            BUG_CHECK(false, "Unsupported builtin");
        }
    }
    else if (auto* externFunc = instance->to<P4::ExternFunction>()) {
        auto name = symbols.getSymbol(externFunc->method);

        // TODO: overloaded function names

        auto* p4RetType = externFunc->method->type->returnType;
        if (!p4RetType || p4RetType->is<IR::Type_Void>()) {
            // P4 mlir does not have a void type, the call in this case has 0 return types
            builder.create<p4mlir::CallOp>(loca, name, operands);
        } else {
            auto retType = toMLIRType(builder, p4RetType);
            auto callOp = builder.create<p4mlir::CallOp>(loca, retType, name, operands);
            valuesTable.addUnchecked(call, callOp.getResult(0));
        }
    }

    // P4 has copy-in/copy-out semantics for calls.
    // At this point written stack allocated variables must be copied back into
    // its original memory
    std::for_each(args->begin(), args->end(), [&](const IR::Argument* arg) {
        // Only variables that were passed as p4.ref<T> must be copied back
        mlir::Value argVal = valuesTable.getUnchecked(arg);
        if (!isRef(argVal)) {
            return;
        }
        mlir::Value tmpAddr = valuesTable.getAddr(arg);
        auto type = tmpAddr.getType().cast<p4mlir::RefType>().getType();
        auto tmpVal = builder.create<p4mlir::LoadOp>(loc(builder, arg), type, tmpAddr);
        auto addr = valuesTable.getAddr(arg->expression);
        builder.create<p4mlir::StoreOp>(loc(builder, arg), addr, tmpVal);
    });
}

void MLIRGenImplCFG::postorder(const IR::Argument* arg) {
    // P4 has copy-in/copy-out semantics for calls.
    // Stack allocated variables must be copied into temporaries before a call,
    // and copied back after. The 'copy back' part is done while visiting
    // `MethodCallExpression`
    mlir::Value exprValue = valuesTable.getUnchecked(arg->expression);

    // A register allocated value is immutable, no need to copy them into temporaries.
    // Stack allocated variables passed as read-only arguments are already materialized
    // at this point
    if (!isRef(exprValue)) {
        valuesTable.add(arg, exprValue);
        return;
    }

    // Stack allocated variable passed as a writeable argument must be copied into
    // a temporary stack space first
    auto refType = exprValue.getType();
    auto type = refType.cast<p4mlir::RefType>().getType();
    mlir::Value tmpAddr = builder.create<p4mlir::AllocOp>(loc(builder, arg), refType);

    // Gets the direction of the parameter to which this argument is bound to
    auto getArgDir = [&]() -> std::optional<IR::Direction> {
        auto* ctxt = getContext();
        if (!ctxt) {
            return std::nullopt;
        }
        auto* call = ctxt->parent->node->to<IR::MethodCallExpression>();
        if (!call) {
            return std::nullopt;
        }
        auto* type = call->method->type->to<IR::Type_MethodBase>();
        if (!type) {
            return std::nullopt;
        }
        auto* param = type->parameters->getParameter(ctxt->child_index);
        if (!param) {
            return std::nullopt;
        }
        return param->direction;
    };

    // Initialize the allocated stack space.
    // Only inout stack arguments must be initialized
    auto dir = getArgDir();
    BUG_CHECK(dir.has_value(), "Could not retrieve argument direction");
    BUG_CHECK(dir == IR::Direction::InOut || dir == IR::Direction::Out,
              "At this point, read-only arguments must be already handled");
    if (dir == IR::Direction::InOut) {
        mlir::Value tmpVal = builder.create<p4mlir::LoadOp>(loc(builder, arg), type, exprValue);
        builder.create<p4mlir::StoreOp>(loc(builder, arg), tmpAddr, tmpVal);
    }

    valuesTable.addAddr(arg, tmpAddr);
}

void MLIRGenImplCFG::postorder(const IR::PathExpression* pe) {
    auto* type = typeMap->getType(pe, true);
    if (!isPrimitiveType(type)) {
        // TODO: this does not consider extern, block and table references
        return;
    }

    // References of SSA values do not generate any operations
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path, true);
    if (allocation.get(decl) == AllocType::REG) {
        // If the SSA value is read, associate it with this 'PathExpression'.
        // It simplifies retrieving the value later
        if (isRead()) {
            ID ssaID = ssaInfo.getID(pe);
            valuesTable.add(pe, valuesTable.get(decl, ssaID));
        }
        return;
    }

    // Retrieve the address of the stack variable
    BUG_CHECK(allocation.get(decl) == AllocType::STACK, "Expected STACK allocation");
    mlir::Value addr = valuesTable.getAddr(decl);

    // Do not materialize the value if it will be written
    if (isWrite()) {
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Even if only read, do not materialize a composite value only because of a member access
    BUG_CHECK(!findContext<IR::ArrayIndex>(), "Not implemented");
    if (findContext<IR::Member>()) {
        valuesTable.addAddr(pe, addr);
        return;
    }

    // Materialize the value in a read context
    BUG_CHECK(isRead(), "Value must be within a read context");
    auto refType = addr.getType().cast<p4mlir::RefType>();
    auto val = builder.create<p4mlir::LoadOp>(loc(builder, pe), refType.getType(), addr);
    valuesTable.add(pe, val);
}

void MLIRGenImplCFG::postorder(const IR::StructExpression* se) {
    // Allocate stack space for the result of the struct expression
    CHECK_NULL(se->structType);
    auto* typeName = se->structType->checkedTo<IR::Type_Name>();
    CHECK_NULL(typeName->path);
    auto* p4type = refMap->getDeclaration(typeName->path, true)->to<IR::Type>();
    auto type = toMLIRType(builder, p4type);
    auto refType = wrappedIntoRef(builder, type);
    auto loca = loc(builder, se);
    mlir::Value addr = builder.create<p4mlir::AllocOp>(loca, refType);

    // Initialize the fields of the allocated stack space.
    // The initializers are represented as {fieldName, expression} pairs
    auto& namedExprs = se->components;
    std::for_each(namedExprs.begin(), namedExprs.end(), [&](const IR::NamedExpression* namedExpr) {
        auto fieldName = builder.getStringAttr(namedExpr->getName().toString().c_str());
        auto fieldType = toMLIRType(builder, typeMap->getType(namedExpr->expression, true));
        auto fieldRefType = wrappedIntoRef(builder, fieldType);
        auto loca = loc(builder, se);
        auto ref = builder.create<p4mlir::GetMemberRefOp>(loca, fieldRefType, addr, fieldName);
        mlir::Value value = valuesTable.get(namedExpr->expression);
        builder.create<p4mlir::StoreOp>(loca, ref, value);
    });

    // P4 states that a header initialized by a list expression has its validity bit set to `true`.
    // MLIR represents this bit as an explicit '__valid' member field, that must be now initialized
    // explicitly
    if (type.isa<p4mlir::HeaderType>()) {
        auto fieldName = builder.getStringAttr("__valid");
        auto fieldRefType = wrappedIntoRef(builder, builder.getIntegerType(1));
        auto ref = builder.create<p4mlir::GetMemberRefOp>(loca, fieldRefType, addr, fieldName);
        mlir::Value cst = builder.create<p4mlir::ConstantOp>(loca, builder.getIntegerType(1), true);
        builder.create<p4mlir::StoreOp>(loca, ref, cst);
    }

    // Always materialize the value, since `StructExpression` cannot be used in a write context
    mlir::Value val = builder.create<p4mlir::LoadOp>(loca, type, addr);
    valuesTable.add(se, val);
}

void MLIRGenImplCFG::postorder(const IR::Add* add) {
    handleArithmeticOp<p4mlir::AddOp>(add);
}

void MLIRGenImplCFG::postorder(const IR::Sub* sub) {
    handleArithmeticOp<p4mlir::SubOp>(sub);
}

void MLIRGenImplCFG::postorder(const IR::Mul* mul) {
    handleArithmeticOp<p4mlir::MulOp>(mul);
}

bool MLIRGenImplCFG::preorder(const IR::IfStatement* ifStmt) {
    visit(ifStmt->condition);
    mlir::Value cond = valuesTable.get(ifStmt->condition);
    mlir::Block* tBlock = getMLIRBlock(currBlock->getTrueSuccessor());
    mlir::Block* fBlock = getMLIRBlock(currBlock->getFalseSuccessor());
    auto tArgs =
        createBlockArgs(ssaInfo, currBlock, currBlock->getTrueSuccessor(), valuesTable);
    auto fArgs =
        createBlockArgs(ssaInfo, currBlock, currBlock->getFalseSuccessor(), valuesTable);
    auto l = loc(builder, ifStmt);
    builder.create<mlir::cf::CondBranchOp>(l, cond, tBlock, tArgs, fBlock, fArgs);
    return false;
}

mlir::Block* MLIRGenImplCFG::getMLIRBlock(const BasicBlock* p4block) const {
    BUG_CHECK(blocksTable.count(p4block), "Could not retrieve corresponding MLIR block");
    return blocksTable.at(p4block);
}

template <typename OpType>
void MLIRGenImplCFG::handleArithmeticOp(const IR::Operation_Binary* arithOp) {
    // Check that 'OpType' has 'SameOperandsAndResultType' trait
    static_assert(OpType::template hasTrait<::mlir::OpTrait::SameOperandsAndResultType>());

    mlir::Value lValue = valuesTable.get(arithOp->left);
    mlir::Value rValue = valuesTable.get(arithOp->right);

    auto val = builder.create<OpType>(loc(builder, arithOp), lValue, rValue);
    valuesTable.add(arithOp, val);
}

std::optional<mlir::Value> MLIRGenImplCFG::getSelfValue() const {
    return selfValue;
}

bool MLIRGenImpl::preorder(const IR::P4Control* control) {
    // Collect apply/ctr parameters (the order is important)
    auto* applyParams = control->getApplyParameters();
    auto* ctrParams = control->getConstructorParameters();
    std::vector<const IR::Parameter*> allParams;
    std::copy(applyParams->begin(), applyParams->end(), std::back_inserter(allParams));
    std::copy(ctrParams->begin(), ctrParams->end(), std::back_inserter(allParams));

    // Given parameter, returns MLIR type of the parameter
    auto MLIRtype = [&](const IR::Parameter* param) {
        auto type = toMLIRType(builder, typeMap->getType(param, true));
        auto dir = param->direction;
        if (dir == IR::Direction::Out || dir == IR::Direction::InOut) {
            type = wrappedIntoRef(builder, type);
        }
        return type;
    };

    // Given collection of parameters, returns MLIR types of those parameters
    auto MLIRtypes = [&](const auto& params) {
        std::vector<mlir::Type> types;
        std::transform(params.begin(), params.end(), std::back_inserter(types), MLIRtype);
        return types;
    };

    // Create ControlOp with 1 block
    llvm::StringRef name(control->getName().toString());
    auto ctrFuncType = builder.getFunctionType(MLIRtypes(*ctrParams), mlir::TypeRange{});
    auto applyFuncType = builder.getFunctionType(MLIRtypes(*applyParams), mlir::TypeRange{});
    auto controlOp =
        builder.create<p4mlir::ControlOp>(loc(builder, control), name, applyFuncType, ctrFuncType);
    auto saved = builder.saveInsertionPoint();
    auto& block = controlOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&block);

    // Add P4 apply/ctr parameters as MLIR block parameters
    std::for_each(allParams.begin(), allParams.end(), [&](const IR::Parameter* param) {
        auto arg = block.addArgument(MLIRtype(param), loc(builder, control));
        if (allocation.get(param) == AllocType::REG) {
            ID ssaID = ssaInfo.getID(param);
            valuesTable.add(param, ssaID, arg);
        } else {
            valuesTable.addAddr(param, arg);
        }
    });

    // Generate everything apart from the apply method and out-of-apply local declarations
    auto& locals = control->controlLocals;
    std::for_each(locals.begin(), locals.end(), [&](const IR::Declaration* decl) {
       auto* type = typeMap->getType(decl, true);
       if (!isPrimitiveType(type)) {
           visit(decl);
       }
    });

    // Create ApplyOp with 1 block
    auto applyOp = builder.create<p4mlir::ApplyOp>(loc(builder, control));
    auto& applyBlock = applyOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&applyBlock);

    // Generate the apply method (which includes out-of-apply local declarations)
    BUG_CHECK(cfgInfo.contains(control), "Could not find CFG for control block");
    genMLIRFromCFG(control, cfgInfo.get(control), applyOp.getBody());

    builder.restoreInsertionPoint(saved);

    // Everything must be generated from within this function, do not traverse further
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Action* action) {
    // Create ActionOp with 1 block
    llvm::StringRef name(action->getName().toString());
    auto funcType = toMLIRType(builder, typeMap->getType(action, true));
    auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name,
                                                  funcType.cast<mlir::FunctionType>());
    auto& block = actOp.getBody().emplaceBlock();
    auto saved = builder.saveInsertionPoint();

    // Add P4 parameters as MLIR block parameters and bind them with their P4 counterparts
    auto* params = action->getParameters();
    std::for_each(params->begin(), params->end(), [&](const IR::Parameter* param) {
        auto type = toMLIRType(builder, typeMap->getType(param, true));
        auto dir = param->direction;
        if (dir == IR::Direction::None || dir == IR::Direction::In) {
            BUG_CHECK(allocation.get(param) == AllocType::REG, "Expected REG allocation");
            auto arg = block.addArgument(type, loc(builder, action));
            ID ssaID = ssaInfo.getID(param);
            valuesTable.add(param, ssaID, arg);
        } else {
            BUG_CHECK(allocation.get(param) == AllocType::STACK, "Expected STACK allocation");
            type = wrappedIntoRef(builder, type);
            auto arg = block.addArgument(type, loc(builder, action));
            valuesTable.addAddr(param, arg);
        }
    });

    // Find enclosing control block (does not have to exist)
    BlockContext context = findContext<IR::P4Control>();

    // Generate action body
    genMLIRFromCFG(context, cfgInfo.get(action), actOp.getBody());

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Method* method) {
    // Create ExternOp
    llvm::StringRef name(method->getName().toString());
    auto funcType = toMLIRType(builder, typeMap->getType(method, true));
    auto actOp = builder.create<p4mlir::ExternOp>(loc(builder, method), name,
                                                  funcType.cast<mlir::FunctionType>());
    return false;
}

void MLIRGenImpl::genMLIRFromCFG(BlockContext context, CFG cfg, mlir::Region& targetRegion) {
    BUG_CHECK(targetRegion.hasOneBlock(), "Target region must have the entry block");
    auto saved = builder.saveInsertionPoint();

    // For each BasicBlock create MLIR Block and insert it into the 'targetRegion'
    CFGWalker::controlFlowTraversal(cfg.getEntry(), [&](BasicBlock* bb) {
        mlir::Block* block = nullptr;
        if (bb == cfg.getEntry()) {
            // MLIR block for the entry block is already created
            block = &targetRegion.front();
        } else {
            block = &targetRegion.emplaceBlock();
        }
        blocksTable.insert({bb, block});
    });

    // For each mlir::Block insert block arguments using phi nodes info.
    // These block arguments must be then bound to {decl, ssa id} pairs,
    // so that references can be resolved during MLIRgen
    CFGWalker::preorder(cfg.getEntry(), [&](BasicBlock* bb) {
        mlir::Block* block = blocksTable.at(bb);
        auto phiInfo = ssaInfo.getPhiInfo(bb);
        for (auto& [decl, phi] : phiInfo) {
            auto loc = builder.getUnknownLoc();
            auto type = toMLIRType(builder, typeMap->getType(decl->to<IR::Declaration>(), true));
            BUG_CHECK(allocation.get(decl) == AllocType::REG,
                      "Unexpected allocation for a block argument");
            mlir::BlockArgument arg = block->addArgument(type, loc);
            // Bind the arg
            ID ssaID = phi.destination.value();
            valuesTable.add(decl, ssaID, arg);
        }
    });

    // Optionally create `self` value which is used to access member variables
    std::optional<mlir::Value> selfValue;
    if (context) {
        BUG_CHECK(blocksTable.count(cfg.getEntry()), "Could not retrieve entry MLIR block");
        auto* block = blocksTable.at(cfg.getEntry());
        builder.setInsertionPointToStart(block);
        const IR::Type* p4type = context.getType();
        auto contextType = toMLIRType(builder, p4type);
        auto refType = wrappedIntoRef(builder, contextType);
        selfValue = builder.create<p4mlir::SelfOp>(loc(builder, context.toNode()), refType);
    }

    // Fill the MLIR Blocks with Ops
    auto* cfgMLIRGen = createCFGVisitor(selfValue);
    CFGWalker::preorder(cfg.getEntry(), [&](BasicBlock* bb) {
        BUG_CHECK(blocksTable.count(bb), "Could not retrieve MLIR block");
        auto* block = blocksTable.at(bb);
        builder.setInsertionPointToEnd(block);
        auto& comps = bb->components;
        std::for_each(comps.begin(), comps.end(), [&](auto* node) {
            cfgMLIRGen->apply(node, bb);
        });
    });

    // Terminate blocks which had no terminator in CFG.
    // If the block has 1 successor insert BranchOp.
    // If the block has no successors insert ReturnOp.
    CFGWalker::preorder(cfg.getEntry(), [&](BasicBlock* bb) {
        BUG_CHECK(blocksTable.count(bb), "Could not retrive MLIR block");
        auto* block = blocksTable.at(bb);
        if (!block->empty() && block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            return;
        }
        BUG_CHECK(bb->succs.size() <= 1, "Non-terminated block can have at most 1 successor");
        builder.setInsertionPointToEnd(block);
        auto loc = builder.getUnknownLoc();
        if (bb->succs.size() == 1) {
            auto* succ = bb->succs.front();
            BUG_CHECK(blocksTable.count(succ), "Could not retrive MLIR block");
            auto args = createBlockArgs(ssaInfo, bb, succ, valuesTable);
            builder.create<mlir::cf::BranchOp>(loc, blocksTable.at(succ), args);
        } else {
            builder.create<p4mlir::ReturnOp>(loc);
        }
    });

    builder.restoreInsertionPoint(saved);
}

MLIRGenImplCFG* MLIRGenImpl::createCFGVisitor(std::optional<mlir::Value> selfValue) {
    return new MLIRGenImplCFG(builder, valuesTable, allocation, blocksTable, typeMap, refMap,
                              ssaInfo, selfValue, symbols);
}

bool MLIRGenImpl::preorder(const IR::Type_Header* hdr) {
    // Create HeaderOp and insert 1 block
    cstring name = hdr->name;
    auto hdrOp = builder.create<p4mlir::HeaderOp>(loc(builder, hdr), llvm::StringRef(name.c_str()));
    auto saved = builder.saveInsertionPoint();
    auto& block = hdrOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    // Generate member declarations
    visit(hdr->fields);

    // Generate validity bit member declaration, which must be the last member
    builder.create<p4mlir::ValidBitDeclOp>(loc(builder, hdr), "__valid");

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Type_Struct* str) {
    // Create StructOp and insert 1 block
    cstring name = str->name;
    auto strOp = builder.create<p4mlir::StructOp>(loc(builder, str), llvm::StringRef(name.c_str()));
    auto saved = builder.saveInsertionPoint();
    auto& block = strOp.getBody().emplaceBlock();
    builder.setInsertionPointToEnd(&block);

    // Generate member declarations
    visit(str->fields);

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::StructField* field) {
    auto* p4type = field->type;
    if (auto* typeName = p4type->to<IR::Type_Name>()) {
        CHECK_NULL(typeName->path);
        p4type = refMap->getDeclaration(typeName->path, true)->to<IR::Type>();
    }
    auto type = toMLIRType(builder, p4type);
    cstring name = field->name;
    builder.create<p4mlir::MemberDeclOp>(loc(builder, field), llvm::StringRef(name), type);
    return false;
}

bool MLIRGenImpl::preorder(const IR::Declaration_Variable* decl) {
    if (decl->initializer) {
        BUG_CHECK(false, "Not implemented");
    }
    auto type = toMLIRType(builder, typeMap->getType(decl, true));
    cstring name = decl->getName();
    builder.create<p4mlir::MemberDeclOp>(loc(builder, decl), llvm::StringRef(name), type);
    return false;
}

mlir::OwningOpRef<mlir::ModuleOp>
mlirGen(mlir::MLIRContext& context, const IR::P4Program* program) {
    mlir::OpBuilder builder(&context);
    auto moduleOp = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(moduleOp.getBody());

    MLIRGen gen(builder);
    program->apply(gen);

    if (!program || ::errorCount() > 0) {
        return nullptr;
    }
    if (failed(mlir::verify(moduleOp))) {
        // Dump for debugging purposes
        moduleOp->print(llvm::outs());
        moduleOp.emitError("module verification error");
        return nullptr;
    }

    return moduleOp;
}


} // namespace p4mlir