#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_

#include <exception>
#include <iterator>
#include <optional>
#include <stack>
#include <variant>

#include "cfg.h"
#include "domTree.h"
#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"
#include "ir/dump.h"
#include "ir/ir.h"
#include "ir/visitor.h"
#include "lib/log.h"
#include "lib/ordered_map.h"
#include "lib/ordered_set.h"
#include "utils.h"

namespace p4mlir {

// TODO: rename this, possibly remove
bool isPrimitiveType(const IR::Type *type);

// Gathers all referenced variables which need an allocation.
class GatherAllocatableVariables : public Inspector
{
    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

    // Declarations of all referenced variables
    ordered_set<const IR::IDeclaration*> vars;

 public:
    GatherAllocatableVariables(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
        : refMap(refMap_), typeMap(typeMap_) {
        CHECK_NULL(refMap, typeMap);
    }

    ordered_set<const IR::IDeclaration*> getReferencedVars() const { return vars; }

 private:
    bool preorder(const IR::Declaration_Instance* decl) override;
    bool preorder(const IR::Declaration_Variable* decl) override;
    bool preorder(const IR::Declaration_Constant* decl) override;
    bool preorder(const IR::P4Table* decl) override;
    bool preorder(const IR::Parameter* param) override;
};

enum class AllocType { REG, STACK, EXTERN, EXTERN_MEMBER, CONSTANT_MEMBER };

// Container to hold allocation types for all allocatable variables
class Allocation
{
    ordered_map<const IR::IDeclaration*, AllocType> data;

 public:
    // Sets variables that need allocation. By default assigns REG allocation
    void setAllocatableVariables(const ordered_set<const IR::IDeclaration*>& allocatable);

    // Assign 'allocType' allocation do variable 'decl'
    void set(const IR::IDeclaration* decl, AllocType allocType);

    // Return assigned allocation type for variable 'decl'.
    // Asserts that the allocation exists
    AllocType get(const IR::IDeclaration* decl) const;

    // Returns all variables which have been assigned allocation type 'type'
    ordered_set<const IR::IDeclaration*> getAllOf(AllocType type) const;
};

// Assigns 'AllocType' to allocatable variables.
// The allocation depends on the type and context of the references
class AllocateVariables : public Inspector, P4WriteContext
{
    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

    // Assigned allocations. Output of this pass
    Allocation& allocation;

 public:
    AllocateVariables(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_,
                      Allocation &allocation_)
        : refMap(refMap_), typeMap(typeMap_), allocation(allocation_) {
        CHECK_NULL(refMap, typeMap);
    }

 private:
    profile_t init_apply(const IR::Node* node) override;
    void end_apply(const IR::Node *root) override;
    bool preorder(const IR::Parameter* param) override;
    bool preorder(const IR::PathExpression* pe) override;
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::P4Parser* control) override;
    bool preorder(const IR::P4Table* table) override;
    bool preorder(const IR::Declaration_Instance* decl) override;
    bool preorder(const IR::Declaration_Constant* decl) override;
};

// Convenience class to hold additional info about references
struct RefInfo {
    std::variant<const IR::IDeclaration*, const IR::PathExpression*> ref;
    // decl == ref for declarations
    const IR::IDeclaration* decl;
};

// This pass is meant to be applied on a single statement.
// Given allocation of variables, gathers written and read REG variables
class GatherSSAReferences : public Inspector, P4WriteContext
{
    // Output of this pass
    std::vector<RefInfo> reads;
    std::vector<RefInfo> writes;

    const P4::TypeMap* typeMap;
    const P4::ReferenceMap* refMap;

    // Output of 'AllocateVariables' pass.
    // Is used to determine REG references
    const Allocation& allocation;

 public:
   GatherSSAReferences(const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                       const Allocation& allocation_)
       : typeMap(typeMap_), refMap(refMap_), allocation(allocation_) {}

   std::vector<RefInfo> getReads() const { return reads; }
   std::vector<RefInfo> getWrites() const { return writes; }

 private:
    bool preorder(const IR::PathExpression *pe) override;
    bool preorder(const IR::Declaration* decl) override;
    bool preorder(const IR::IfStatement* ifStmt) override;
    bool preorder(const IR::SwitchStatement* switchStmt) override;
    bool preorder(const IR::NamedExpression*) override { return true; };

 private:
    void addRead(const IR::PathExpression* pe, const IR::IDeclaration* decl);
    void addWrite(const IR::PathExpression* pe, const IR::IDeclaration* decl);
    void addWrite(const IR::IDeclaration* decl);
};

// ID used for SSA numbering
using ID = std::size_t;

// Container to hold numbering info of a single phi node
struct PhiInfo
{
    std::optional<ID> destination;
    ordered_map<const BasicBlock*, std::optional<ID>> sources;
};

// Stores the result of the SSA conversion, i.e. phi nodes placement and numbering of ssa values
class SSAInfo
{
    // For each basic block stores its phi nodes.
    // Each phi node belongs to a variable (IR::IDeclaration).
    // Phi node for var V looks like this:
    //      V = phi(V, ..., V)
    // 1 argument for each predecessor.
    ordered_map<const BasicBlock *, ordered_map<const IR::IDeclaration *, PhiInfo>> phiInfo;

    // Stores ID for each use/def of an SSA value
    ordered_map<std::variant<const IR::IDeclaration *, const IR::PathExpression *>, ID> ssaRefIDs;

 public:
    void addPhi(const BasicBlock *bb, const IR::IDeclaration *var);
    void numberRef(ID id, std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref);
    void numberPhiDestination(ID id, const BasicBlock *block, const IR::IDeclaration *var);
    void numberPhiSource(ID id, const BasicBlock *block, const IR::IDeclaration *var,
                         const BasicBlock *source);

 public:
    // Returns calculated phi nodes info for block 'bb'.
    // Return value states for which variables there exists a phi node and what numbering was
    // calculated for phi node arguments and destination
    ordered_map<const IR::IDeclaration *, PhiInfo> getPhiInfo(const BasicBlock *bb) const;

    // Convenience method to check if 'bb' contains phi node for variable 'var'
    bool phiExists(const BasicBlock *bb, const IR::IDeclaration *var) const;

    // Returns calculated number of as SSA value reference.
    // Asserts the SSA numbering exists
    ID getID(std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) const;
};

// Performs SSA conversion algorithm.
// Given allocation and CFGs, calculates SSA form of the whole program, numbering all REG allocated
// references
class MakeSSAInfo : public Inspector
{
    // Output of this pass
    SSAInfo& ssaInfo;

    const CFGInfo& cfgInfo;
    const Allocation& allocation;
    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

 public:
    MakeSSAInfo(SSAInfo &ssaInfo_, const CFGInfo &cfgInfo_, const Allocation &allocation_,
                const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
        : ssaInfo(ssaInfo_),
          cfgInfo(cfgInfo_),
          allocation(allocation_),
          refMap(refMap_),
          typeMap(typeMap_) {
        CHECK_NULL(refMap, typeMap);
    }

 private:
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::P4Parser* parser) override;
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::ParserState* state) override;
    bool preorder(const IR::PathExpression* pe) override;
    bool preorder(const IR::Declaration_Instance* decl) override;
    bool preorder(const IR::Declaration_Constant* decl) override;

 private:
    // Calculates SSA form of the 'cfg' control flow graph.
    // 'params' must contain all REG allocated variables which are referenced within the 'cfg'
    // but are not declared within the 'cfg'. Those include action parameters and apply/ctr
    // parameters
    void traverseCFG(const ordered_set<const IR::IDeclaration*>& params, CFG cfg);

    void rename(const BasicBlock* block,
                ordered_map<const IR::IDeclaration*, ID>& nextIDs,
                ordered_map<const IR::IDeclaration*, std::stack<ID>>& stkIDs,
                const DomTree* domTree) const;

};


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */