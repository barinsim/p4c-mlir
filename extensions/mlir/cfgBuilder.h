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

class CFGBuilder : public Inspector 
{
    class Builder 
    {
        BasicBlock* curr = nullptr;
    public:
        std::map<const IR::IFunctional*, BasicBlock*> callableToCFG;
        void add(const IR::StatOrDecl* item);
        void addSuccessor(const BasicBlock* succ);
        void enterBasicBlock(BasicBlock* bb);
    };

    Builder b;
    
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::StatOrDecl* statOrDecl) override;
    bool preorder(const IR::BlockStatement*) override {
        // Should 2 different scopes be in a single basic block?
        // Just ignore for now.
        return true;
    }
    bool preorder(const IR::IfStatement* ifStmt) override;
    
};


} // namespace p4mlir

#endif /* BACKENDS_MLIR_CFGBUILDER_H_ */