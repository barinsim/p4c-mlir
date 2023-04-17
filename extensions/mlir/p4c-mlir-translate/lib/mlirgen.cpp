#include "mlirgen.h"


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
    BUG_CHECK(assign->left->is<IR::PathExpression>(), "Not implemented");
    auto* pe = assign->left->to<IR::PathExpression>();
    auto rValue = toValue(assign->right);

    // Write to an SSA register allocated variable
    if (ssaInfo.isSSARef(pe)) {
        mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
        addValue(assign->left, value);
        return;
    }

    // Write to a stack allocated variable
    mlir::Value addr = toValue(pe);
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
    // TODO: this needs to consider if the object is in reg or stack
    auto type = toMLIRType(builder, typeMap->getType(mem));
    auto name = builder.getStringAttr(mem->member.toString().c_str());
    mlir::Value val =
        builder.create<p4mlir::GetMemberOp>(loc(builder, mem), type, toValue(mem->expr), name);
    addValue(mem, val);
}

void MLIRGenImplCFG::postorder(const IR::MethodCallExpression* call) {
    // Resolve name of the callable
    CHECK_NULL(call->method);
    BUG_CHECK(call->method->is<IR::PathExpression>(), "Not implemented");
    auto* expr = call->method->to<IR::PathExpression>();
    CHECK_NULL(expr->path);
    auto* decl = refMap->getDeclaration(expr->path);
    llvm::StringRef name(decl->getName().toString().c_str());

    // Get result MLIR type
    mlir::TypeRange results = {};

    // Resolve call arguments to MLIR values
    std::vector<mlir::Value> operands;
    auto* args = call->arguments;
    std::transform(args->begin(), args->end(), std::back_inserter(operands),
                   [&](const IR::Argument *arg) { return toValue(arg); });

    // Insert the call operation
    auto callOp =
        builder.create<p4mlir::CallOp>(loc(builder, call), results, name, operands);
    BUG_CHECK(callOp.getNumResults() == 0, "Not implemented");

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

    // Get the value containing the reference of the stack allocated variable
    CHECK_NULL(pe->path);
    auto* decl = refMap->getDeclaration(pe->path);
    mlir::Value addr = toValue(decl);
    BUG_CHECK(addr.getType().isa<p4mlir::RefType>(), "Stack allocated variable without address");
    auto refType = addr.getType().cast<p4mlir::RefType>();

    // Do not materialize the value if it will be written
    if (isWrite()) {
        addValue(pe, addr);
        return;
    }

    // Materialize the value in a read context
    auto val = builder.create<p4mlir::LoadOp>(loc(builder, pe), refType.getType(), addr);
    addValue(pe, val);
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

bool MLIRGenImpl::preorder(const IR::P4Control* control) {
    // Create ControlOp with 1 block
    llvm::StringRef name(control->getName().toString());
    auto controlOp = builder.create<p4mlir::ControlOp>(loc(builder, control), name);
    auto saved = builder.saveInsertionPoint();
    auto& block = controlOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&block);

    // Add P4 apply parameters as MLIR block parameters
    auto* applyParams = control->getApplyParameters();
    std::for_each(applyParams->begin(), applyParams->end(), [&](const IR::Declaration* param) {
        auto type = toMLIRType(builder, typeMap->getType(param));
        block.addArgument(type, loc(builder, control));
    });

    // Generate everything within control block apart from the apply method
    visit(control->controlLocals);

    // Generate the apply method
    auto applyOp = builder.create<p4mlir::ApplyOp>(loc(builder, control->body));
    genMLIRFromCFG(control->body, applyOp.getBody());

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::P4Action* action) {
    // Create ActionOp
    llvm::StringRef name(action->getName().toString());
    auto actOp = builder.create<p4mlir::ActionOp>(loc(builder, action), name);
    auto saved = builder.saveInsertionPoint();

    // Generate action body
    genMLIRFromCFG(action, actOp.getBody());

    builder.restoreInsertionPoint(saved);
    return false;
}

void MLIRGenImpl::genMLIRFromCFG(const IR::Node* decl, mlir::Region& targetRegion) {
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
    auto* context = getCurrentNode<IR::IApply>();
    if (!context) {
        context = findContext<IR::IApply>();
    }
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
            auto type = toMLIRType(builder, typeMap->getType(decl->to<IR::Declaration>()));
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

    // Add mapping of the outer apply parameters, so that references of these parameters can be
    // resolved. These MLIR parameters must already be created
    if (context) {
        mlir::Region* parent = targetRegion.getParentRegion();
        auto blockParams = parent->getArguments();
        auto* applyParams = context->getApplyParameters();
        BUG_CHECK(blockParams.size() == applyParams->size(),
                  "Number of P4 apply parameters and MLIR block parameters must match");
        for (int i = 0; i < blockParams.size(); ++i) {
            auto* applyParam = applyParams->getParameter(i);
            BUG_CHECK(ssaInfo.isSSARef(applyParam), "Could not find SSA info for the parameter");
            auto id = ssaInfo.getID(applyParam);
            SSARefType ref{applyParam, id};
            ssaRefMap.insert({ref, blockParams[i]});
        }
    }

    // Fill the MLIR Blocks with Ops
    MLIRGenImplCFG cfgGen(builder, mapping, typeMap, refMap, ssaInfo, ssaRefMap);
    CFGWalker::preorder(entry, [&](BasicBlock* bb) {
        BUG_CHECK(mapping.count(bb), "Could not retrieve MLIR block");
        auto* block = mapping.at(bb);
        builder.setInsertionPointToStart(block);
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

    builder.restoreInsertionPoint(saved);
    return false;
}

bool MLIRGenImpl::preorder(const IR::StructField* field) {
    auto type = toMLIRType(builder, field->type);
    cstring name = field->name;
    builder.create<p4mlir::MemberDeclOp>(loc(builder, field), llvm::StringRef(name), type);
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