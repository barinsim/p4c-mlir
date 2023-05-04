#include "mlirgen.h"

#include "frontends/p4/methodInstance.h"


namespace p4mlir {

namespace {

mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node) {
    // TODO:
    CHECK_NULL(node);
    return mlir::FileLineColLoc::get(builder.getStringAttr("test/file.p4"), 42, 422);
}

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

mlir::Type wrappedIntoRef(mlir::OpBuilder& builder, mlir::Type type) {
    BUG_CHECK(!type.isa<RefType>(), "Ref type cannot be wrapped into another reference");
    return p4mlir::RefType::get(builder.getContext(), type);
}

std::vector<mlir::Value> createBlockArgs(const SSAInfo &ssaInfo, const BasicBlock *bb,
                                         const BasicBlock *succ,
                                         const std::map<SSARefType, mlir::Value> &refMap) {
    CHECK_NULL(bb, succ);
    std::vector<mlir::Value> rv;
    auto phiInfo = ssaInfo.getPhiInfo(succ);
    for (auto &[decl, phi] : phiInfo) {
        BUG_CHECK(phi.sources.count(bb), "Phi node does not contain argument for the block");
        auto id = phi.sources.at(bb).value();
        SSARefType ref{decl, id};
        BUG_CHECK(refMap.count(ref), "Could not resolve phi argument into value");
        auto argVal = refMap.at(ref);
        rv.push_back(argVal);
    }
    return rv;
}

} // namespace

void MLIRGenImplCFG::postorder(const IR::BoolLiteral* boolean) {
    auto type = toMLIRType(builder, typeMap->getType(boolean));
    CHECK_NULL(type);
    mlir::Value val = builder.create<p4mlir::ConstantOp>(loc(builder, boolean), type,
                                                          (int64_t)boolean->value);
    addValue(boolean, val);
}

void MLIRGenImplCFG::postorder(const IR::Constant* cst) {
    auto type = toMLIRType(builder, typeMap->getType(cst));
    CHECK_NULL(type);
    BUG_CHECK(cst->fitsInt64(), "Not implemented");
    mlir::Value val =
        builder.create<p4mlir::ConstantOp>(loc(builder, cst), type, cst->asInt64());
    addValue(cst, val);
}

void MLIRGenImplCFG::postorder(const IR::ReturnStatement* ret) {
    builder.create<p4mlir::ReturnOp>(loc(builder, ret));
}

void MLIRGenImplCFG::postorder(const IR::AssignmentStatement* assign) {
    auto rValue = toValue(assign->right);

    // Write to an SSA register allocated variable
    auto* pe = assign->left->to<IR::PathExpression>();
    if (pe && ssaInfo.isSSARef(pe)) {
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
        addValue(assign->left, value);
        return;
    }

    // Write to a stack allocated variable
    mlir::Value addr = toValue(assign->left);
    BUG_CHECK(addr.getType().isa<p4mlir::RefType>(), "Stack allocated variable without an address");
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
        return toValue(decl->initializer);
    };

    auto init = createInitValue(decl);

    // No need to allocate space for reg variables
    if (ssaInfo.isSSARef(decl)) {
        auto value = init;
        if (decl->initializer) {
            auto type = toMLIRType(builder, typeMap->getType(decl));
            value = builder.create<p4mlir::CopyOp>(loc(builder, decl), type, init);
        }
        addValue(decl, value);
        return;
    }

    // Create space for stack allocated variables
    auto type = toMLIRType(builder, typeMap->getType(decl));
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr = builder.create<p4mlir::AllocOp>(loc(builder, decl), refType);
    builder.create<p4mlir::StoreOp>(loc(builder, decl), addr, init);
    addValue(decl, addr);
}

void MLIRGenImplCFG::postorder(const IR::Cast* cast) {
    CHECK_NULL(cast->destType);
    auto src = toValue(cast->expr);
    auto targetType = toMLIRType(builder, cast->destType);
    mlir::Value value = builder.create<p4mlir::CastOp>(loc(builder, cast), targetType, src);
    addValue(cast, value);
}

void MLIRGenImplCFG::postorder(const IR::Operation_Relation* cmp) {
    auto lhsValue = toValue(cmp->left);
    auto rhsValue = toValue(cmp->right);

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
    addValue(cmp, res);
}

void MLIRGenImplCFG::postorder(const IR::Member* mem) {
    // Callable reference does not generate any ops, skip it
    auto p4type = typeMap->getType(mem, true);
    if (p4type->is<IR::Type_Method>()) {
        return;
    }

    auto type = toMLIRType(builder, p4type);
    auto name = builder.getStringAttr(mem->member.toString().c_str());
    mlir::Value baseValue = toValue(mem->expr);
    mlir::Type baseType = baseValue.getType();

    // Materialize member of a register allocated variable
    if (!baseType.isa<p4mlir::RefType>()) {
        BUG_CHECK(isRead(),
                  "Member access to a register variable can be used only in a read context");
        mlir::Value val =
            builder.create<p4mlir::GetMemberOp>(loc(builder, mem), type, baseValue, name);
        addValue(mem, val);
        return;
    }

    // Retrieve member reference of a stack allocated variable
    auto refType = wrappedIntoRef(builder, type);
    mlir::Value addr =
        builder.create<p4mlir::GetMemberRefOp>(loc(builder, mem), refType, baseValue, name);

    // Member variable is written, return just the reference
    if (isWrite()) {
        addValue(mem, addr);
        return;
    }

    // Member variable is only read.
    // If this is the last member access within the path, materialize the value.
    // Otherwise return just a reference
    if (findContext<IR::Member>()) {
        addValue(mem, addr);
        return;
    }
    mlir::Value val = builder.create<p4mlir::LoadOp>(loc(builder, mem), type, addr);
    addValue(mem, val);
}

void MLIRGenImplCFG::postorder(const IR::MethodCallExpression* call) {
    // Resolve call arguments to MLIR values
    std::vector<mlir::Value> operands;
    auto* args = call->arguments;
    std::transform(args->begin(), args->end(), std::back_inserter(operands),
                   [&](const IR::Argument *arg) { return toValue(arg); });

    // 'MethodCallExpression' represents different types of calls, each of which
    // needs to generate different ops.
    // Figure out which call this is and generate correct mlir
    auto* instance = P4::MethodInstance::resolve(call, refMap, typeMap);
    auto loca = loc(builder, call);
    CHECK_NULL(instance);

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
            auto refType = wrappedIntoRef(builder, type);
            auto fieldName = builder.getStringAttr("__valid");
            if (base.getType().isa<p4mlir::RefType>()) {
                return builder.create<p4mlir::GetMemberRefOp>(loca, refType, base, fieldName);
            }
            return builder.create<p4mlir::GetMemberOp>(loca, type, base, fieldName);
        };

        auto member = getValidBit(toValue(builtin->appliedTo));

        // Translate builtins of the header type into explicit operations on the __valid field
        if (name == "setValid" || name == "setInvalid") {
            auto cst = builder.create<p4mlir::ConstantOp>(loca, type, name == "setValid");
            builder.create<p4mlir::StoreOp>(loca, member, cst);
        } else if (name == "isValid") {
            mlir::Value value = member;
            if (member.getType().isa<p4mlir::RefType>()) {
                value = builder.create<p4mlir::LoadOp>(loca, type, member);
            }
            addValue(call, value);
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
            addValue(call, callOp.getResult(0));
        }
    }

    // P4 has copy-in/copy-out semantics for calls.
    // At this point written stack allocated variables must be copied back into
    // its original memory
    std::for_each(args->begin(), args->end(), [&](const IR::Argument* arg) {
        // Only variables that were passed as p4.ref<T> must be copied back
        if (!toValue(arg).getType().isa<p4mlir::RefType>()) {
            return;
        }
        mlir::Value tmpAddr = toValue(arg);
        auto type = tmpAddr.getType().cast<p4mlir::RefType>().getType();
        auto tmpVal = builder.create<p4mlir::LoadOp>(loc(builder, arg), type, tmpAddr);
        auto addr = toValue(arg->expression);
        builder.create<p4mlir::StoreOp>(loc(builder, arg), addr, tmpVal);
    });
}

void MLIRGenImplCFG::postorder(const IR::Argument* arg) {
    // P4 has copy-in/copy-out semantics for calls.
    // Stack allocated variables must be copied into temporaries before a call,
    // and copied back after. The 'copy back' part is done while visiting
    // `MethodCallExpression`

    mlir::Value exprValue = toValue(arg->expression);

    // A register allocated value is immutable, no need to copy them into temporaries.
    // Stack allocated variables passed as read-only arguments are already materialized
    // at this point
    if (!exprValue.getType().isa<p4mlir::RefType>()) {
        addValue(arg, exprValue);
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

    addValue(arg, tmpAddr);
}

void MLIRGenImplCFG::postorder(const IR::PathExpression* pe) {
    // References of SSA values do not generate any operations
    auto* type = typeMap->getType(pe, true);
    if (ssaInfo.isSSARef(pe) || !isPrimitiveType(type)) {
        return;
    }

    // Gets the address of a stack local variable or a member variable
    auto getAddress = [&]() {
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);

        // Member variable address must be loaded through the `self` reference
        if (members.count(decl)) {
            std::optional<mlir::Value> self = getSelfValue();
            BUG_CHECK(self.has_value(),
                      "Member variable is referenced but the `self` reference does not exist");
            auto mlirType = toMLIRType(builder, type);
            auto refType = wrappedIntoRef(builder, mlirType);
            auto name = builder.getStringAttr(decl->getName().toString().c_str());
            mlir::Value addr =
                builder.create<p4mlir::GetMemberRefOp>(loc(builder, pe), refType, *self, name);
            return addr;
        }

        // The variable is local and is stack allocated.
        // Get the value containing the reference
        return toValue(decl);
    };

    mlir::Value addr = getAddress();
    BUG_CHECK(addr.getType().isa<p4mlir::RefType>(), "Address must have `!p4.ref<T>` type");

    // Do not materialize the value if it will be written
    if (isWrite()) {
        addValue(pe, addr);
        return;
    }

    // Even if only read, do not materialize a composite value only because of a member access
    BUG_CHECK(!findContext<IR::ArrayIndex>(), "Not implemented");
    if (findContext<IR::Member>()) {
        addValue(pe, addr);
        return;
    }

    // Materialize the value in a read context
    BUG_CHECK(isRead(), "Value must be within a read context");
    auto refType = addr.getType().cast<p4mlir::RefType>();
    auto val = builder.create<p4mlir::LoadOp>(loc(builder, pe), refType.getType(), addr);
    addValue(pe, val);
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
        mlir::Value value = toValue(namedExpr->expression);
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
    addValue(se, val);
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
    mlir::Value cond = toValue(ifStmt->condition);
    mlir::Block* tBlock = getMLIRBlock(currBlock->getTrueSuccessor());
    mlir::Block* fBlock = getMLIRBlock(currBlock->getFalseSuccessor());
    auto tArgs =
        createBlockArgs(ssaInfo, currBlock, currBlock->getTrueSuccessor(), ssaRefToValue);
    auto fArgs =
        createBlockArgs(ssaInfo, currBlock, currBlock->getFalseSuccessor(), ssaRefToValue);
    auto l = loc(builder, ifStmt);
    builder.create<mlir::cf::CondBranchOp>(l, cond, tBlock, tArgs, fBlock, fArgs);
    return false;
}

void MLIRGenImplCFG::addValue(const IR::INode* node, mlir::Value value) {
    // Check if 'node' represents SSA value write,
    // through IR::PathExpression or IR::IDeclaration.
    // These must be stored separately to be able to resolve
    // references of these values
    auto* decl = node->to<IR::IDeclaration>();
    if (decl) {
        SSARefType ref{decl, 0};
        ssaRefToValue.insert({ref, value});
        return;
    }
    auto* pe = node->to<IR::PathExpression>();
    if (pe && ssaInfo.isSSARef(pe)) {
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);
        SSARefType ref{decl, ssaInfo.getID(pe)};
        ssaRefToValue.insert({ref, value});
        return;
    }
    nodeToValue.insert({node, value});
}

mlir::Value MLIRGenImplCFG::toValue(const IR::IDeclaration* decl, SSAInfo::ID id) const {
    SSARefType ref{decl, id};
    BUG_CHECK(ssaRefToValue.count(ref), "No matching value found");
    return ssaRefToValue.at(ref);
}

mlir::Value MLIRGenImplCFG::toValue(const IR::INode* node) const {
    // SSA value reference is handled differently.
    // The mlir::Value is not connected directly to the
    // AST node, rather to unique {decl, SSA id} pair
    auto* pe = node->to<IR::PathExpression>();
    if (pe && ssaInfo.isSSARef(pe)) {
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);
        return toValue(decl, ssaInfo.getID(pe));
    }
    auto* decl = node->to<IR::IDeclaration>();
    if (decl) {
        return toValue(decl, 0);
    }
    return nodeToValue.at(node);
}

mlir::Block* MLIRGenImplCFG::getMLIRBlock(const BasicBlock* p4block) const {
    BUG_CHECK(blocksMapping.count(p4block), "Could not retrieve corresponding MLIR block");
    return blocksMapping.at(p4block);
}

template <typename OpType>
void MLIRGenImplCFG::handleArithmeticOp(const IR::Operation_Binary* arithOp) {
    // Check that 'OpType' has 'SameOperandsAndResultType' trait
    static_assert(OpType::template hasTrait<::mlir::OpTrait::SameOperandsAndResultType>());

    mlir::Value lValue = toValue(arithOp->left);
    mlir::Value rValue = toValue(arithOp->right);

    auto val = builder.create<OpType>(loc(builder, arithOp), lValue, rValue);
    addValue(arithOp, val);
}

std::optional<mlir::Value> MLIRGenImplCFG::getSelfValue() const {
    return selfValue;
}

bool MLIRGenImpl::preorder(const IR::P4Control* control) {
    // Collect types of the apply parameters of this control block
    std::vector<mlir::Type> applyTypes;
    auto* applyParams = control->getApplyParameters();
    std::transform(applyParams->begin(), applyParams->end(), std::back_inserter(applyTypes),
                   [&](const IR::Parameter* param) {
                       auto type = toMLIRType(builder, typeMap->getType(param, true));
                       auto dir = param->direction;
                       if (dir == IR::Direction::InOut || dir == IR::Direction::Out) {
                           type = wrappedIntoRef(builder, type);
                       }
                       return type;
                   });

    // Collect types of the constructor parameters of this control block
    std::vector<mlir::Type> ctrTypes;
    auto* ctrParams = control->constructorParams;
    std::transform(ctrParams->begin(), ctrParams->end(), std::back_inserter(ctrTypes),
                   [&](const IR::Declaration* param) {
                       auto* type = typeMap->getType(param, true);
                       return toMLIRType(builder, type);
                   });

    // Create ControlOp with 1 block
    llvm::StringRef name(control->getName().toString());
    auto ctrFuncType = builder.getFunctionType(ctrTypes, mlir::TypeRange{});
    auto applyFuncType = builder.getFunctionType(applyTypes, mlir::TypeRange{});
    auto controlOp =
        builder.create<p4mlir::ControlOp>(loc(builder, control), name, applyFuncType, ctrFuncType);
    auto saved = builder.saveInsertionPoint();
    auto& block = controlOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&block);

    // Add P4 apply parameters as MLIR block parameters
    std::for_each(applyTypes.begin(), applyTypes.end(), [&](mlir::Type type) {
        block.addArgument(type, loc(builder, control));
    });

    // Add P4 constructor parameters as MLIR block parameters
    std::for_each(ctrTypes.begin(), ctrTypes.end(), [&](mlir::Type type) {
        block.addArgument(type, loc(builder, control));
    });

    // Generate everything within control block apart from the apply method
    visit(control->controlLocals);

    // Generate the apply method
    auto applyOp = builder.create<p4mlir::ApplyOp>(loc(builder, control->body));
    genMLIRFromCFG(control, control->body, applyOp.getBody());

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Action* action) {
    // Create ActionOp
    llvm::StringRef name(action->getName().toString());
    auto funcType = toMLIRType(builder, typeMap->getType(action, true));
    auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name,
                                                  funcType.cast<mlir::FunctionType>());
    auto saved = builder.saveInsertionPoint();

    // Find enclosing control block (does not have to exist)
    BlockContext context = findContext<IR::P4Control>();

    // Generate action body
    genMLIRFromCFG(context, action, actOp.getBody());

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

void MLIRGenImpl::genMLIRFromCFG(BlockContext context, const IR::Node *decl,
                                 mlir::Region &targetRegion) {
    CHECK_NULL(decl);
    BUG_CHECK(cfg.count(decl), "Could retrieve CFG for the declaration");
    auto saved = builder.saveInsertionPoint();

    // Retrieve CFG for 'decl'
    BasicBlock* entry = cfg.at(decl);

    // For each BasicBlock create MLIR Block and insert it into the 'targetRegion'
    ordered_map<const BasicBlock*, mlir::Block*> mapping;
    CFGWalker::controlFlowTraversal(entry, [&](BasicBlock* bb) {
        auto& block = targetRegion.emplaceBlock();
        mapping.insert({bb, &block});
    });

    // Create ssa mapping for this cfg
    SSAInfo ssaInfo(context, {decl, entry}, refMap, typeMap);

    // Stores mapping of P4 ssa values to its MLIR counterparts.
    // This mapping is used during MLIRgen to resolve references
    std::map<SSARefType, mlir::Value> ssaRefMap;

    // For each mlir::Block insert block arguments using phi nodes info.
    // These block arguments must be then bound to {decl, ssa id} pairs,
    // so that references can be resolved during MLIRgen
    CFGWalker::preorder(entry, [&](BasicBlock* bb) {
        mlir::Block* block = mapping.at(bb);
        auto phiInfo = ssaInfo.getPhiInfo(bb);
        for (auto& [decl, phi] : phiInfo) {
            auto loc = builder.getUnknownLoc();
            auto type = toMLIRType(builder, typeMap->getType(decl->to<IR::Declaration>(), true));
            if (!ssaInfo.isSSARef(decl)) {
                type = wrappedIntoRef(builder, type);
            }
            mlir::BlockArgument arg = block->addArgument(type, loc);
            // Bind the arg
            auto id = phi.destination.value();
            SSARefType ref{decl, id};
            BUG_CHECK(!ssaRefMap.count(ref), "Binding already exists");
            ssaRefMap.insert({ref, arg});
        }
    });

    // Add mapping of the outer apply/ctr parameters, so that references of these parameters can be
    // resolved. These MLIR parameters must already be created
    if (context) {
        mlir::Region* parent = targetRegion.getParentRegion();
        auto blockParams = parent->getArguments();
        auto* applyParams = context.getApplyParameters();
        auto* ctrParams = context.getConstructorParameters();
        BUG_CHECK(blockParams.size() == applyParams->size() + ctrParams->size(),
                  "Number of P4 parameters and MLIR block parameters must match");
        int blockParamIdx = 0;
        for (auto* params : {applyParams, ctrParams}) {
            for (int i = 0; i < params->size(); ++i) {
                auto* param = params->getParameter(i);
                BUG_CHECK(ssaInfo.isSSARef(param), "Could not find SSA info for the parameter");
                auto id = ssaInfo.getID(param);
                SSARefType ref{param, id};
                auto bParam = blockParams[blockParamIdx++];
                ssaRefMap.insert({ref, bParam});
            }
        }
    }

    // Get member variables of the context block.
    // Context does not have to exist in which case empty set is returned
    auto members = context.getMemberVariables();

    // Optionally create `self` value which is used to access member variables
    std::optional<mlir::Value> selfValue;
    if (context) {
        BUG_CHECK(mapping.count(entry), "Could not retrieve entry MLIR block");
        auto* block = mapping.at(entry);
        builder.setInsertionPointToStart(block);
        const IR::Type* p4type = context.getType();
        auto contextType = toMLIRType(builder, p4type);
        auto refType = wrappedIntoRef(builder, contextType);
        selfValue = builder.create<p4mlir::SelfOp>(loc(builder, decl), refType);
    }

    // Fill the MLIR Blocks with Ops
    MLIRGenImplCFG cfgGen(builder, mapping, typeMap, refMap, ssaInfo, ssaRefMap, members,
                          selfValue, context, symbols);
    CFGWalker::preorder(entry, [&](BasicBlock* bb) {
        BUG_CHECK(mapping.count(bb), "Could not retrieve MLIR block");
        auto* block = mapping.at(bb);
        builder.setInsertionPointToEnd(block);
        auto& comps = bb->components;
        std::for_each(comps.begin(), comps.end(), [&](auto* stmt) {
            cfgGen.apply(stmt, bb);
        });
    });

    // Terminate blocks which had no terminator in CFG.
    // If the block has 1 successor insert BranchOp.
    // If the block has no successors insert ReturnOp.
    CFGWalker::preorder(entry, [&](BasicBlock* bb) {
        BUG_CHECK(mapping.count(bb), "Could not retrive MLIR block");
        auto* block = mapping.at(bb);
        if (!block->empty() && block->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
            return;
        }
        BUG_CHECK(bb->succs.size() <= 1, "Non-terminated block can have at most 1 successor");
        builder.setInsertionPointToEnd(block);
        auto loc = builder.getUnknownLoc();
        if (bb->succs.size() == 1) {
            auto* succ = bb->succs.front();
            BUG_CHECK(mapping.count(succ), "Could not retrive MLIR block");
            auto args = createBlockArgs(ssaInfo, bb, succ, ssaRefMap);
            builder.create<mlir::cf::BranchOp>(loc, mapping.at(succ), args);
        } else {
            builder.create<p4mlir::ReturnOp>(loc);
        }
    });

    builder.restoreInsertionPoint(saved);
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