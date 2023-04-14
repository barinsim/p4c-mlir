#ifndef BACKENDS_MLIR_SSA_H_
#define BACKENDS_MLIR_SSA_H_


#include <variant>
#include <stack>
#include <exception>
#include <optional>
#include <iterator>

#include "ir/ir.h"
#include "ir/visitor.h"
#include "ir/dump.h"

#include "frontends/common/resolveReferences/referenceMap.h"
#include "frontends/p4/typeMap.h"

#include "lib/log.h"
#include "lib/ordered_map.h"
#include "lib/ordered_set.h"

#include "cfgBuilder.h"
#include "domTree.h"


namespace p4mlir {


bool isPrimitiveType(const IR::Type *type);

// Assigns either REG or STACK allocation to all referenced variables.
// The allocation depends on the type and context of the reference.
// If applied multiple times existing REG allocations can be overwritten by STACK,
// but not vice versa. All references of REG variables will be later assigned SSA number
class AllocateVariables : public Inspector, P4WriteContext
{
    // Gathers all referenced variables which need an allocation.
    class GatherAllReferencedVariables : public Inspector
    {
        const P4::ReferenceMap* refMap;
        const P4::TypeMap* typeMap;

        // Declarations of all referenced variables
        ordered_set<const IR::IDeclaration*> vars;

     public:
        GatherAllReferencedVariables(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
            : refMap(refMap_), typeMap(typeMap_) {
            CHECK_NULL(refMap, typeMap);
        }

        ordered_set<const IR::IDeclaration*> getReferencedVars() const { return vars; }

     private:
        bool preorder(const IR::Declaration* decl) override;
        bool preorder(const IR::PathExpression* pe) override;
    };

    const P4::ReferenceMap* refMap;
    const P4::TypeMap* typeMap;

    // Assigned allocation.
    // Contains allocation for all referenced variables
    enum class AllocType { REG, STACK };
    ordered_map<const IR::IDeclaration*, AllocType> allocation;

 public:
    AllocateVariables(const P4::ReferenceMap *refMap_, const P4::TypeMap *typeMap_)
        : refMap(refMap_), typeMap(typeMap_) {
        CHECK_NULL(refMap, typeMap);
    }

    // Gets all variables with REG allocation
    ordered_set<const IR::IDeclaration*> getRegVariables() const;

    // True if 'decl' has REG allocation
    bool isRegVariable(const IR::IDeclaration* decl) const;

 private:
    profile_t init_apply(const IR::Node* node) override;
    bool preorder(const IR::Parameter* param) override;
    bool preorder(const IR::PathExpression* pe) override;

 private:
    // Assigns STACK allocation to 'decl'
    void allocateToStack(const IR::IDeclaration* decl);
};

struct RefInfo {
    std::variant<const IR::IDeclaration*, const IR::PathExpression*> ref;
    // decl == ref for declarations
    const IR::IDeclaration* decl;
};

// This pass is meant to be applied on a single statement.
// Given REG variables from 'AllocateVariables' pass, it determines
// which REG variables are written and which are read within the statement
class GatherSSAReferences : public Inspector, P4WriteContext
{
    class Builder {
        std::vector<RefInfo> reads;
        std::vector<RefInfo> writes;
     public:
        void addRead(const IR::PathExpression* pe, const IR::IDeclaration* decl);
        void addWrite(const IR::PathExpression* pe, const IR::IDeclaration* decl);
        void addWrite(const IR::IDeclaration* decl);
        std::vector<RefInfo> getReads() const { return reads; };
        std::vector<RefInfo> getWrites() const { return writes; };
    };

    Builder b;

    const P4::TypeMap* typeMap;
    const P4::ReferenceMap* refMap;

    // Output of 'AllocateVariables' pass.
    // Other than these variables are ignored by this pass
    ordered_set<const IR::IDeclaration*> ssaVariables;

public:
   GatherSSAReferences(const P4::TypeMap *typeMap_, const P4::ReferenceMap *refMap_,
                       ordered_set<const IR::IDeclaration *> ssaVariables_)
       : typeMap(typeMap_), refMap(refMap_), ssaVariables(std::move(ssaVariables_)) {}

   std::vector<RefInfo> getReads() const { return b.getReads(); }
   std::vector<RefInfo> getWrites() const { return b.getWrites(); }

private:
    bool preorder(const IR::PathExpression *pe) override;
    bool preorder(const IR::Declaration* decl) override;
    bool preorder(const IR::IfStatement* ifStmt) override;
    bool preorder(const IR::SwitchStatement* switchStmt) override;
};

// Given P4 action or apply method, performs SSA conversion algorithm.
// Determines which values are placed into SSA registers and which onto a stack.
// Stores the result of the conversion, i.e. phi nodes placement and numbering of ssa values.
class SSAInfo
{
 public:
    using ID = std::size_t;

    struct Phi {
        std::optional<ID> destination;
        ordered_map<const BasicBlock*, std::optional<ID>> sources;
    };

 private:
    // For each basic block stores its phi nodes.
    // Each phi node belongs to a variable (IR::Declaration).
    // Phi node for var V looks like this:
    //      V = phi(V, ..., V)
    // 1 argument for each predecessor.
    ordered_map<const BasicBlock*, ordered_map<const IR::IDeclaration*, Phi>>
        phiInfo;

    // Stores ID for each use/def of an SSA value
    ordered_map<std::variant<const IR::IDeclaration *, const IR::PathExpression *>, ID>
        ssaRefIDs;

    class Builder {
        decltype(SSAInfo::phiInfo) phiInfo;
        decltype(SSAInfo::ssaRefIDs) ssaRefIDs;

     public:
        void addPhi(const BasicBlock* bb, const IR::IDeclaration* var);
        void numberRef(ID id, std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref);
        void numberPhiDestination(ID id, const BasicBlock* block, const  IR::IDeclaration* var);
        void numberPhiSource(ID id, const BasicBlock *block, const IR::IDeclaration *var,
                             const BasicBlock *source);
        bool phiExists(const BasicBlock* bb, const IR::IDeclaration* var) const;
        ordered_set<const IR::IDeclaration*> getPhiInfo(const BasicBlock* bb) const;

        decltype(phiInfo) movePhiInfo() const { return std::move(phiInfo); }
        decltype(ssaRefIDs) moveRefsInfo() const { return std::move(ssaRefIDs); }
    };

 public:
    // Calculates SSA form. Determines phi nodes positions and numbers P4 references of SSA values.
    // 'context' is used to take apply parameters of the outer block into account, can be null.
    SSAInfo(const IR::IApply* context, std::pair<const IR::Node*, const BasicBlock*> cfg,
            const P4::ReferenceMap* refMap, const P4::TypeMap* typeMap);

    // Returns calculated phi nodes info for block 'bb'.
    // Return value states for which variables there exists a phi node and what numbering was
    // calculated for phi node arguments and destination
    ordered_map<const IR::IDeclaration*, Phi> getPhiInfo(const BasicBlock* bb) const {
        if (phiInfo.count(bb)) {
            return phiInfo.at(bb);
        }
        return {};
    }

    bool isSSARef(std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) const {
        return ssaRefIDs.count(ref);
    }

    ID getID(std::variant<const IR::IDeclaration *, const IR::PathExpression *> ref) const {
        BUG_CHECK(isSSARef(ref), "Reference is not an SSA value");
        return ssaRefIDs.at(ref);
    }

 private:
    void rename(const BasicBlock* block, Builder& b,
                ordered_map<const IR::IDeclaration*, ID>& nextIDs,
                ordered_map<const IR::IDeclaration*, std::stack<ID>>& stkIDs,
                const DomTree* domTree,
                const P4::TypeMap* typeMap,
                const P4::ReferenceMap* refMap,
                const ordered_set<const IR::IDeclaration*>& forbidden) const;
};


} // namespace p4mlir


#endif /* BACKENDS_MLIR_SSA_H_ */