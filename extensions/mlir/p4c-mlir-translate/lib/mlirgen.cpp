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
    return;
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
    // TODO: writes through pointer
    BUG_CHECK(assign->left->is<IR::PathExpression>(), "Not implemented");

    auto rValue = toValue(assign->right);
    mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, assign), rValue);
    addValue(assign->left, value);
}

void MLIRGenImplCFG::postorder(const IR::Declaration_Variable* decl) {
    if (!decl->initializer) {
        auto type = toMLIRType(builder, typeMap->getType(decl));
        mlir::Value value = builder.create<p4mlir::UninitializedOp>(loc(builder, decl), type);
        addValue(decl, value);
        return;
    }
    mlir::Value init = toValue(decl->initializer);
    mlir::Value value = builder.create<p4mlir::CopyOp>(loc(builder, decl), init);
    addValue(decl, value);
}

void MLIRGenImplCFG::postorder(const IR::Cast* cast) {
    CHECK_NULL(cast->destType);
    auto src = toValue(cast->expr);
    auto targetType = toMLIRType(builder, cast->destType);
    auto value = builder.create<p4mlir::CastOp>(loc(builder, cast), targetType, src);
    addValue(cast, value);
}

void MLIRGenImplCFG::postorder(const IR::Equ* eq) {
    auto lhsValue = toValue(eq->left);
    auto rhsValue = toValue(eq->right);
    auto res =
        builder.create<CompareOp>(loc(builder, eq), CompareOpKind::eq, lhsValue, rhsValue);
    addValue(eq, res);
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

void MLIRGenImplCFG::addValue(const IR::Node* node, mlir::Value value) {
    // Check if 'node' represents SSA value write,
    // through IR::PathExpression or IR::IDeclaration.
    // These must be stored separately to be able to resolve
    // references of these values
    auto* decl = node->to<IR::IDeclaration>();
    if (decl && ssaInfo.isSSARef(decl)) {
        SSARefType ref{decl, ssaInfo.getID(decl)};
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
    BUG_CHECK(node->is<IR::Expression>(),
              "Value can be associated only with an expression at this point");
    exprToValue.insert({node->to<IR::Expression>(), value});
}

mlir::Value MLIRGenImplCFG::toValue(const IR::IDeclaration* decl, SSAInfo::ID id) const {
    SSARefType ref{decl, id};
    BUG_CHECK(ssaRefToValue.count(ref), "No matching value found");
    return ssaRefToValue.at(ref);
}

mlir::Value MLIRGenImplCFG::toValue(const IR::Node* node) const {
    // SSA value reference is handled differently.
    // The mlir::Value is not connected directly to the
    // AST node, rather to unique {decl, SSA id} pair
    auto* pe = node->to<IR::PathExpression>();
    if (pe && ssaInfo.isSSARef(pe)) {
        CHECK_NULL(pe->path);
        auto* decl = refMap->getDeclaration(pe->path);
        return toValue(decl, ssaInfo.getID(pe));
    }
    BUG_CHECK(node->is<IR::Expression>(), "At this point node must be an expression");
    return exprToValue.at(node->to<IR::Expression>());
}

mlir::Block* MLIRGenImplCFG::getMLIRBlock(const BasicBlock* p4block) const {
    BUG_CHECK(blocksMapping.count(p4block), "Could not retrieve corresponding MLIR block");
    return blocksMapping.at(p4block);
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
            mlir::BlockArgument arg = block->addArgument(type, loc);
            // Bind the arg
            auto id = phi.destination.value();
            SSARefType ref{decl, id};
            BUG_CHECK(!ssaRefMap.count(ref), "Binding already exists");
            ssaRefMap.insert({ref, arg});
        }
    });

    // Add mapping of the outter apply parameters, so that references of these parameters can be
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
        BUG_CHECK(mapping.count(bb), "Could not retrive MLIR block");
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

bool MLIRGenImpl::preorder(const IR::Header* hdr) {
    //mlir::TypeSubElementReplacements
    return true;
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
      moduleOp.emitError("module verification error");
      return nullptr;
    }

    return moduleOp;
}


} // namespace p4mlir