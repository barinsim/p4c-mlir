#include "frontends/common/resolveReferences/resolveReferences.h"
#include "frontends/p4/typeChecking/typeChecker.h"
#include "frontends/p4/typeMap.h"

#include "ir/dump.h"
#include "ir/ir.h"
#include "ir/pass_manager.h"
#include "ir/visitor.h"

#include "lib/ordered_map.h"

#include "P4Dialect.h"
#include "P4Ops.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "ssa.h"
#include "utils.h"
#include "cfg.h"

namespace p4mlir {

// Class faciliating conversion of the P4 types into MLIR types
class TypeConvertor
{
    OpBuilder& builder;
    P4::TypeMap* typeMap = nullptr;
    P4::ReferenceMap* refMap = nullptr;

 public:
    TypeConvertor(OpBuilder& builder_, P4::TypeMap *typeMap_, P4::ReferenceMap *refMap_)
        : builder(builder_), typeMap(typeMap_), refMap(refMap_) {
        CHECK_NULL(typeMap, refMap);
    }

    // Converts P4 type into the corresponding MLIR type
    mlir::Type toMLIRType(const IR::Type* p4type) const;
};

// Stores mapping of P4 value references to MLIR values
class ValuesTable
{
    using ExprOrArg = std::variant<const IR::Expression*, const IR::Argument*>;
    using SSARef = std::pair<const IR::IDeclaration*, ID>;
    using StackVar = const IR::IDeclaration*;

    // Represents all types that can be associated with MLIR value
    using BindableType = std::variant<StackVar, ExprOrArg, SSARef>;

    ordered_map<BindableType, mlir::Value> data;

 public:
    void add(ExprOrArg expr, mlir::Value value) {
        BUG_CHECK(!value.getType().isa<p4mlir::RefType>(), "Expected value type");
        addUnchecked(expr, value);
    };
    void add(const IR::IDeclaration* decl, ID id, mlir::Value value) {
        BUG_CHECK(!value.getType().isa<p4mlir::RefType>(), "Expected value type");
        addUnchecked(SSARef{decl, id}, value);
    };
    void addAddr(const IR::IDeclaration* decl, mlir::Value addr) {
        BUG_CHECK(addr.getType().isa<p4mlir::RefType>(), "Expected ref type");
        addUnchecked(decl, addr);
    };
    void addAddr(ExprOrArg expr, mlir::Value addr) {
        BUG_CHECK(addr.getType().isa<p4mlir::RefType>(), "Expected ref type");
        addUnchecked(expr, addr);
    };
    void addUnchecked(BindableType node, mlir::Value addr) {
        BUG_CHECK(!data.count(node), "Value for a node already exists");
        data.insert({node, addr});
    };
    mlir::Value get(ExprOrArg expr) const {
        auto val = getUnchecked(expr);
        BUG_CHECK(!val.getType().isa<p4mlir::RefType>(), "Expected value type");
        return val;
    };
    mlir::Value get(const IR::IDeclaration* decl, ID id) const {
        auto val = getUnchecked(SSARef{decl, id});
        BUG_CHECK(!val.getType().isa<p4mlir::RefType>(), "Expected value type");
        return val;
    };
    mlir::Value getAddr(const IR::IDeclaration* decl) const {
        auto val = getUnchecked(decl);
        BUG_CHECK(val.getType().isa<p4mlir::RefType>(), "Expected ref type");
        return val;
    };
    mlir::Value getAddr(ExprOrArg expr) const {
        auto val = getUnchecked(expr);
        BUG_CHECK(val.getType().isa<p4mlir::RefType>(), "Expected ref type");
        return val;
    };
    mlir::Value getUnchecked(BindableType node) const {
        BUG_CHECK(data.count(node), "Could not find value for a node");
        return data.at(node);
    };

};

// Visitor used to convert P4 constructs representable via
// Control Flow Graph (i.e. actions, apply methods, parser) into MLIR.
// This class must be applied direclty on the components of CFGBuilder::CFGType.
// Components are visited in the order of execution.
// While visiting terminators (e.g. IR::IfStatement) it is made sure
// that children blocks are not visited.
// The object of this class is meant to be alive for the whole MLIRgen of the entire CFG
class MLIRGenImplCFG : public Inspector, P4WriteContext
{
    mlir::OpBuilder& builder;

    // Block of the currently visited statement
    BasicBlock* currBlock = nullptr;

    // Mapping of P4 SSA values to its MLIR counterparts.
    // Stores both real P4 values and MLIR block parameters
    ValuesTable& valuesTable;

    // Stores allocation types of P4 values
    const Allocation& allocation;

    // Maps P4 AST block to its MLIR counterpart
    const ordered_map<const BasicBlock*, mlir::Block*>& blocksTable;

    // Stores types of expressions
    P4::TypeMap* typeMap = nullptr;

    // Stores mapping of P4 references to its P4 declarations.
    // Does not take SSA into account
    P4::ReferenceMap* refMap = nullptr;

    // Calculated SSA form of the whole program
    const SSAInfo& ssaInfo;

    // Value containing the `self` reference of the parser or control block.
    // Is used for a member variable access.
    // Does not have to exist (e.g. action outside a control block)
    std::optional<mlir::Value> selfValue;

    // Stores a fully qualified symbols for all P4 AST nodes that can be referenced by
    // an MLIR symbol. All symbol references in P4 dialect are assumed to be fully qualified within
    // the parent ModuleOp
    const FullyQualifiedSymbols& symbols;

    // Stores additional parameters which must be explicitly stated in P4 dialect but not in P4.
    // e.g. out-of-apply local variables are converted to explicit table apply parameters
    const AdditionalParams& additionalParams;

    // Internal flag that makes sure this visitor was started properly
    // using custom 'apply' method instead of the usual `node->apply(visitor)`
    bool customApplyCalled = false;

 public:
    MLIRGenImplCFG(mlir::OpBuilder &builder_, ValuesTable &valuesTable_,
                   const Allocation &allocation_,
                   const ordered_map<const BasicBlock *, mlir::Block *> &blocksTable_,
                   P4::TypeMap *typeMap_, P4::ReferenceMap *refMap_, const SSAInfo &ssaInfo_,
                   std::optional<mlir::Value> selfValue_, const FullyQualifiedSymbols &symbols_,
                   const AdditionalParams &additionalParams_)
        : builder(builder_),
          valuesTable(valuesTable_),
          allocation(allocation_),
          blocksTable(blocksTable_),
          typeMap(typeMap_),
          refMap(refMap_),
          ssaInfo(ssaInfo_),
          selfValue(selfValue_),
          symbols(symbols_),
          additionalParams(additionalParams_) {
        CHECK_NULL(typeMap, refMap);
    }

    // This is the main way to run this visitor, 'stmt' must be part of 'bb's components,
    // otherwise a terminator will not be able to map the successors onto MLIR blocks properly.
    // If 'bb' is nullptr, asserts no branching terminators are visited (useful for generating
    // arguments for instantiations)
    void apply(const IR::Node* node, BasicBlock* bb = nullptr) {
        CHECK_NULL(node);
        currBlock = bb;
        customApplyCalled = true;
        node->apply(*this);
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
    void postorder(const IR::Argument* arg) override;
    void postorder(const IR::PathExpression* pe) override;
    void postorder(const IR::StructExpression* se) override;
    void postorder(const IR::ListExpression* le) override;
    void postorder(const IR::DefaultExpression* de) override;

    // --- Arithmetic Operators ---
    void postorder(const IR::Add* add) override;
    void postorder(const IR::Sub* sub) override;
    void postorder(const IR::Mul* mul) override;

    // --- Terminators ---
    void postorder(const IR::ReturnStatement* ret) override;
    bool preorder(const IR::IfStatement* ifStmt) override;

 private:
    // Returns MLIR counterpart of the P4 BasicBlock
    mlir::Block* getMLIRBlock(const BasicBlock* p4block) const;

    // Inserts 'OpType' operation using operands of 'arithOp'.
    // 'OpType' must have 'SameOperandsAndResultType' trait
    template <typename OpType>
    void handleArithmeticOp(const IR::Operation_Binary* arithOp);

    // Gets the value containing the `self` reference
    std::optional<mlir::Value> getSelfValue() const;

    // Converts P4 type into corresponding MLIR type
    mlir::Type toMLIRType(const IR::Type* p4type) const;

    // Builds TransitionOp out of 'SelectExpression' or 'PathExpression' in the case of
    // unconditional transition
    mlir::Operation* buildTransitionOp(
        std::variant<const IR::SelectExpression*, const IR::PathExpression*> node);
};

// Visitor converting valid P4 AST into P4 MLIR dialect
class MLIRGenImpl : public Inspector
{
    mlir::OpBuilder& builder;

    // See 'valuesTable' in 'MLIRGenImplCFG'
    ValuesTable valuesTable;

    // See 'blocksTable' in 'MLIRGenImplCFG'
    ordered_map<const BasicBlock*, mlir::Block*> blocksTable;

    P4::TypeMap* typeMap = nullptr;
    P4::ReferenceMap* refMap = nullptr;

    // Control Flow Graph for all P4 constructs representable by a CFG
    const CFGInfo& cfgInfo;

    // See 'symbols' in 'MLIRGenImplCFG'
    const FullyQualifiedSymbols& symbols;

    // See 'ssaInfo' in 'MLIRGenImplCFG'
    const SSAInfo& ssaInfo;

    // See 'allocation' in 'MLIRGenImplCFG'
    const Allocation& allocation;

    // See 'additionalParams' in 'MLIRGenImplCFG'
    const AdditionalParams& additionalParams;

 public:
    MLIRGenImpl(mlir::OpBuilder &builder_, P4::TypeMap *typeMap_, P4::ReferenceMap *refMap_,
                const CFGInfo &cfgInfo_, const FullyQualifiedSymbols &symbols_,
                const SSAInfo &ssaInfo_, const Allocation &allocation_,
                const AdditionalParams &additionalParams_)
        : builder(builder_),
          typeMap(typeMap_),
          refMap(refMap_),
          cfgInfo(cfgInfo_),
          symbols(symbols_),
          ssaInfo(ssaInfo_),
          allocation(allocation_),
          additionalParams(additionalParams_) {
        CHECK_NULL(typeMap, refMap);
    }

 private:
    bool preorder(const IR::P4Parser* parser) override;
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::ParserState* state) override;
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::Method* method) override;
    bool preorder(const IR::Type_Header* hdr) override;
    bool preorder(const IR::Type_Struct* str) override;
    bool preorder(const IR::Type_Extern* ext) override;
    bool preorder(const IR::StructField* field) override;
    bool preorder(const IR::Declaration* decl) override;
    bool preorder(const IR::P4Table* table) override;
    bool preorder(const IR::Property* property) override;
    bool preorder(const IR::ExpressionValue* exprVal) override;
    bool preorder(const IR::ActionList* actionsList) override;
    bool preorder(const IR::ActionListElement* actionElement) override;
    bool preorder(const IR::EntriesList* entriesList) override;
    bool preorder(const IR::Entry* entry) override;
    bool preorder(const IR::Key* keyList) override;
    bool preorder(const IR::KeyElement* key) override;
    bool preorder(const IR::ExpressionListValue* exprList) override;
    bool preorder(const IR::Declaration_MatchKind* matchKinds) override;

    // Generates MLIR for CFG of 'decl', MLIR blocks are inserted into 'targetRegion'.
    // CFG must be accessible through `cfg['decl']`
    void genMLIRFromCFG(P4Block context, CFG cfg, mlir::Region& targetRegion);

    // Convenience method to create 'MLIRGenImplCFG' from private member variables
    MLIRGenImplCFG* createCFGVisitor(std::optional<mlir::Value> selfValue);

    // Given P4Block 'block' representing P4Control or P4Parser, generates MLIR for either of them.
    // In P4 dialect control and parser are treated very similarly and thus its build process can
    // be unified
    mlir::Operation* buildControlOrParser(P4Block block);

    // Given 'expr' builds TableActionOp including its body.
    // TableActionOp is needed to be built from multiple different AST nodes.
    // This simplifies the process
    TableActionOp buildTableActionOp(const IR::Expression* expr);

    // Converts P4 type into corresponding MLIR type
    mlir::Type toMLIRType(const IR::Type* p4type) const;

    // Given P4Block 'context', generates p4.self op and returns the generated MLIR value.
    // If 'context' represents empty context, returns std::nullopt.
    std::optional<mlir::Value> generateSelfValue(mlir::Location loc, mlir::OpBuilder &builder,
                                                 P4Block context);

    // Given 'ParserState' builds unconditional transition to the state. This is useful if the
    // transition is synthesized
    mlir::Operation* buildTransitionOp(const IR::ParserState* state);
};

// Synthesize 'accept' and 'reject' parser states. This is currently needed to make the type
// checking work
class AddAcceptAndRejectStates : public Transform
{
    const IR::Node *postorder(IR::P4Parser *parser) override {
        parser->states.push_back(new IR::ParserState(IR::ParserState::accept, nullptr));
        parser->states.push_back(new IR::ParserState(IR::ParserState::reject, nullptr));
        return parser;
    }
};

// Main pass of the P4 AST -> P4 MLIR dialect translation
class MLIRGen : public PassManager
{
 public:
    MLIRGen(mlir::OpBuilder& builder) {
        auto* refMap = new P4::ReferenceMap();
        auto* typeMap = new P4::TypeMap();
        auto* additionalParams = new AdditionalParams();
        auto* cfgInfo = new CFGInfo();
        auto* symbols = new FullyQualifiedSymbols();
        auto* allocation = new Allocation();
        auto* ssaInfo = new SSAInfo();
        passes.push_back(new AddAcceptAndRejectStates());
        passes.push_back(new P4::ResolveReferences(refMap));
        passes.push_back(new P4::TypeInference(refMap, typeMap, false, true));
        passes.push_back(new P4::TypeChecking(refMap, typeMap, true));
        passes.push_back(new MakeCFGInfo(*cfgInfo));
        passes.push_back(new AllocateVariables(refMap, typeMap, *allocation));
        passes.push_back(new MakeSSAInfo(*ssaInfo, *cfgInfo, *allocation, refMap, typeMap));
        passes.push_back(new CollectAdditionalParams(*additionalParams));
        passes.push_back(new MakeFullyQualifiedSymbols(builder, *symbols, typeMap));
        passes.push_back(new MLIRGenImpl(builder, typeMap, refMap, *cfgInfo, *symbols, *ssaInfo,
                                         *allocation, *additionalParams));
    }
};

// Main API to convert P4 AST into P4 MLIR dialect.
// P4 dialect must be already registered into 'context'.
// 'program' must be an AST of a valid P4 program
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, const IR::P4Program *program);

} // namespace p4mlir
