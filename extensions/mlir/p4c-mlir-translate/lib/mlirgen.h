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

// Type representing P4 SSA value.
// Consists of P4 AST declaration and id computed during SSA calculation
using SSARefType = std::pair<const IR::IDeclaration *, p4mlir::SSAInfo::ID>;

namespace {

// Converts P4 location stored in 'loc' into its MLIR counterpart
mlir::Location loc(mlir::OpBuilder& builder, const IR::Node* node);

// Converts P4 type into corresponding MLIR type
mlir::Type toMLIRType(mlir::OpBuilder& builder, const IR::Type* p4type);

// Creates block arguments for jump from 'bb' to 'succ'.
// The order of arguments corresponds to phi arguments stored within 'ssaInfo'.
// 'ssaInfo' should be also used to create block parameters to match the order
std::vector<mlir::Value> createBlockArgs(const SSAInfo &ssaInfo, const BasicBlock *bb,
                                         const BasicBlock *succ,
                                         const std::map<SSARefType, mlir::Value> &refMap);

} // namespace

// Visitor used to convert P4 constructs representable via
// Control Flow Graph (i.e. actions, apply methods, parser) into MLIR.
// This class must be applied direclty on the components of CFGBuilder::CFGType.
// Components are visited in the order of execution.
// While visiting terminators (e.g. IR::IfStatement) it is made sure
// that children blocks are not visited.
// The object of this class is meant to be alive for the whole MLIRgen of the entire CFG
class MLIRGenImplCFG : public Inspector
{
    mlir::OpBuilder& builder;

    // Block of the currently visited statement
    BasicBlock* currBlock = nullptr;

    // Mapping of expressions to the MLIR values they produced
    ordered_map<const IR::Expression*, mlir::Value> exprToValue;

    // Mapping of P4 SSA values to its MLIR counterparts.
    // Stores both real P4 values and MLIR block parameters
    std::map<SSARefType, mlir::Value>& ssaRefToValue;

    // Maps P4 AST block to its MLIR counterpart
    const ordered_map<const BasicBlock*, mlir::Block*>& blocksMapping;

    // Stores types of expressions
    const P4::TypeMap* typeMap = nullptr;

    // Stores mapping of P4 references to its P4 declarations.
    // Does not take SSA into account
    const P4::ReferenceMap* refMap = nullptr;

    // SSA mapping of the currently visited CFG.
    // Stores phi functions and SSA value numbering
    const SSAInfo& ssaInfo;

    // Internal flag that makes sure this visitor was started properly
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
    void postorder(const IR::AssignmentStatement* assign) override;
    void postorder(const IR::Declaration_Variable* decl) override;
    void postorder(const IR::Cast* cast) override;
    void postorder(const IR::Operation_Relation* cmp) override;
    void postorder(const IR::Member* eq) override;
    void postorder(const IR::MethodCallExpression* call) override;

    // --- Terminators ---
    void postorder(const IR::ReturnStatement* ret) override;
    bool preorder(const IR::IfStatement* ifStmt) override;

 private:
    // Creates binding between 'node' and 'value'.
    // These bindings are later queried via 'toValue()' and
    // used to resolve SSA value references and results of expressions
    void addValue(const IR::Node* node, mlir::Value value);

    // Some SSA value references do not have corresponding AST node.
    // For example references within phi nodes.
    // These and other values can be retrieved directly via 'decl' and its SSA 'id'
    mlir::Value toValue(const IR::IDeclaration* decl, SSAInfo::ID id) const;

    // Returns value that was previously bound with 'node' via 'addValue()'
    mlir::Value toValue(const IR::Node* node) const;

    // Returns MLIR counterpart of the P4 BasicBlock
    mlir::Block* getMLIRBlock(const BasicBlock* p4block) const;

};

// Visitor converting valid P4 AST into P4 MLIR dialect
class MLIRGenImpl : public Inspector
{
    mlir::OpBuilder& builder;

    const P4::TypeMap* typeMap = nullptr;
    const P4::ReferenceMap* refMap = nullptr;

    // Control Flow Graph for all of the P4 constructs representable by a CFG
    const CFGBuilder::CFGType& cfg;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, const P4::TypeMap *typeMap_,
                const P4::ReferenceMap *refMap_, const CFGBuilder::CFGType &cfg_)
        : builder(builder_), typeMap(typeMap_), refMap(refMap_), cfg(cfg_) {
        CHECK_NULL(typeMap, refMap);
    }

 private:
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::Type_Header* hdr) override;
    bool preorder(const IR::StructField* field) override;

    // Generates MLIR for CFG of 'decl', MLIR blocks are inserted into 'targetRegion'.
    // CFG must be accessible through `cfg['decl']`
    void genMLIRFromCFG(const IR::Node* decl, mlir::Region& targetRegion);

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

// Main API to convert P4 AST into P4 MLIR dialect.
// P4 dialect must be already registered into 'context'.
// 'program' must be an AST of a valid P4 program
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);


} // namespace p4mlir