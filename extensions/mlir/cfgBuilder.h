#ifndef BACKENDS_MLIR_CFGBUILDER_H_
#define BACKENDS_MLIR_CFGBUILDER_H_

#include <vector>
#include <map>

#include "ir/ir.h"
#include "ir/visitor.h"

namespace p4mlir {


struct BasicBlock {
    std::vector<const IR::StatOrDecl*> components;
    std::vector<const BasicBlock*> succs;
    static int nextId;
    int id = nextId++;
};

std::string toString(const BasicBlock* bb, int indent = 0);

class CFGBuilder : public Inspector 
{
 public:
    std::map<const IR::IDeclaration*, BasicBlock*> getCFG() const {
        return b.callableToCFG;
    }

 private:
    class Builder 
    {
        BasicBlock* curr = nullptr;
    public:
        std::map<const IR::IDeclaration*, BasicBlock*> callableToCFG;
        void add(const IR::StatOrDecl* item);
        void addSuccessor(const BasicBlock* succ);
        void enterBasicBlock(BasicBlock* bb);
    };

    Builder b;

    Visitor::profile_t init_apply(const IR::Node *node) override {
        b = Builder();
        return Inspector::init_apply(node);
    }
    
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::IfStatement* ifStmt) override;
    bool preorder(const IR::ReturnStatement* ret) override { b.add(ret); return true; }
    bool preorder(const IR::AssignmentStatement* assign) override { b.add(assign); return true; }
    bool preorder(const IR::MethodCallStatement* call) override { b.add(call); return true; }

    bool preorder(const IR::BlockStatement*) override {
        // Should 2 different scopes be in a single basic block?
        // Just ignore for now.
        return true;
    }
    bool preorder(const IR::ParameterList*) override { return false; }
    bool preorder(const IR::P4Table*) override { return false; }
    bool preorder(const IR::Annotations*) override { return false; }
    bool preorder(const IR::P4Parser*) override { return false; }
    bool preorder(const IR::Attribute*) override { return false; }
    bool preorder(const IR::StructField*) override { return false; }
    bool preorder(const IR::Type_StructLike*) override { return false; }
    
};


class CFGPrinter
{
 public:
    std::string toString(const BasicBlock* entry, int indent = 0) const;

 private:
    void toStringImpl(const BasicBlock* bb, int indent,
                            std::unordered_set<const BasicBlock*>& visited,
                            std::ostream& os) const;
    std::string toString(const IR::Node* node) const;
    std::string getBlockIdentifier(const BasicBlock* bb) const;

};


} // namespace p4mlir

#endif /* BACKENDS_MLIR_CFGBUILDER_H_ */