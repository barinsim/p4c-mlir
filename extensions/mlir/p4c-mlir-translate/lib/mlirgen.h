#include "ir/visitor.h"
#include "ir/ir.h"
#include "ir/pass_manager.h"
#include "ir/dump.h"

#include "lib/ordered_map.h"

#include "frontends/p4/typeMap.h"
#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeChecking/typeChecker.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Block.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "cfgBuilder.h"
#include "ssa.h"

#include "P4Dialect.h"
#include "P4Ops.h"


namespace p4mlir {

using SSARefType = std::pair<const IR::IDeclaration *, p4mlir::SSAInfo::ID>;

namespace {

mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node);

mlir::Type toMLIRType(mlir::OpBuilder& builder, const IR::Type* p4type);

// Creates block arguments for jump from 'bb' to 'succ'.
// The order of arguments corresponds to phi arguments stored within 'ssaInfo'.
// 'ssaInfo' should be also used to create block parameters to match the order.
std::vector<mlir::Value> createBlockArgs(const SSAInfo &ssaInfo, const BasicBlock *bb,
                                         const BasicBlock *succ,
                                         const std::map<SSARefType, mlir::Value> &refMap);

} // namespace


class MLIRGenImplCFG : public Inspector
{
    mlir::OpBuilder& builder;

    BasicBlock* currBlock = nullptr;

    ordered_map<const IR::Expression*, mlir::Value> exprToValue;

    std::map<SSARefType, mlir::Value>& ssaRefToValue;

    const ordered_map<const BasicBlock*, mlir::Block*>& blocksMapping;

    const P4::TypeMap* typeMap = nullptr;
    const P4::ReferenceMap* refMap = nullptr;

    const SSAInfo& ssaInfo;

    // This is an internal flag that makes sure this visitor was started properly
    // using custom 'apply' method instead of the usual `node->apply(visitor)`
    bool customApplyCalled = false;

 public:
    MLIRGenImplCFG(mlir::OpBuilder &builder_,
                   const ordered_map<const BasicBlock *, mlir::Block *> &blocksMapping_,
                   const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                   const SSAInfo& ssaInfo_,
                   std::map<SSARefType, mlir::Value>& ssaRefToValue_)
        : builder(builder_),
          blocksMapping(blocksMapping_),
          typeMap(typeMap_),
          refMap(refMap_),
          ssaInfo(ssaInfo_),
          ssaRefToValue(ssaRefToValue_) {
        CHECK_NULL(typeMap, refMap);
    }

    // This is the main way to run this visitor, 'stmt' must be part of 'bb's components,
    // otherwise a terminator will not be able to map the successors onto MLIR blocks properly
    void apply(const IR::StatOrDecl* stmt, BasicBlock* bb) {
        CHECK_NULL(stmt, bb);
        currBlock = bb;
        customApplyCalled = true;
        stmt->apply(*this);
        customApplyCalled = false;
    }

 private:
    Visitor::profile_t init_apply(const IR::Node *node) override {
        BUG_CHECK(customApplyCalled, "Visitor was not started properly");
        return Inspector::init_apply(node);
    }

    void postorder(const IR::BoolLiteral* boolean) override;
    void postorder(const IR::Constant* cst) override;
    void postorder(const IR::ReturnStatement* ret) override;
    void postorder(const IR::AssignmentStatement* assign) override;
    void postorder(const IR::Declaration_Variable* decl) override;
    void postorder(const IR::Cast* cast) override;
    void postorder(const IR::Equ* eq) override;

    bool preorder(const IR::IfStatement* ifStmt) override;

 private:
    void addValue(const IR::Node* node, mlir::Value value);

    // Some SSA value references do not have corresponding AST node.
    // For example references within phi nodes.
    // These values can be retrived directly via 'decl' and its SSA 'id'
    mlir::Value toValue(const IR::IDeclaration* decl, SSAInfo::ID id) const;

    mlir::Value toValue(const IR::Node* node) const;
    mlir::Block* getMLIRBlock(const BasicBlock* p4block) const;

};

class MLIRGenImpl : public Inspector
{
    mlir::OpBuilder& builder;

    const P4::TypeMap* typeMap = nullptr;
    const P4::ReferenceMap* refMap = nullptr;
    const CFGBuilder::CFGType& cfg;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_,
                const P4::ReferenceMap *refMap_, const CFGBuilder::CFGType &cfg_)
        : builder(builder_), typeMap(typeMap_), refMap(refMap_), cfg(cfg_) {
        CHECK_NULL(typeMap, refMap);
    }

 private:
    bool preorder(const IR::P4Action* action) override;

};

class MLIRGen : public PassManager
{
 public:
    MLIRGen(mlir::OpBuilder& builder) {
        auto* refMap = new P4::ReferenceMap();
        auto* typeMap = new P4::TypeMap();
        auto* cfg = new CFGBuilder::CFGType();
        passes.push_back(new P4::ResolveReferences(refMap));
        passes.push_back(new P4::TypeInference(refMap, typeMap, false, true));
        passes.push_back(new P4::TypeChecking(refMap, typeMap, true));
        passes.push_back(new CFGBuilder(*cfg));
        passes.push_back(new MLIRGenImpl(builder, typeMap, refMap, *cfg));
    }
};


mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir