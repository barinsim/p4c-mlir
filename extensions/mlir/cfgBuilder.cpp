#include "cfgBuilder.h"


namespace p4mlir {


int BasicBlock::nextId = 0;

bool CFGBuilder::preorder(const IR::P4Action* action) {
        const IR::IFunctional* callable = action;
        BUG_CHECK(!b.callableToCFG.count(callable), "");
        BasicBlock* entryBlock = new BasicBlock();
        b.enterBasicBlock(entryBlock);
        b.callableToCFG.insert({callable, entryBlock});
        return true;
    }

bool CFGBuilder::preorder(const IR::StatOrDecl* statOrDecl) {
    b.add(statOrDecl);
    return true;
}

bool CFGBuilder::preorder(const IR::IfStatement* ifStmt) {
    b.add(ifStmt);
    BasicBlock* trueB = new BasicBlock();
    BasicBlock* falseB = new BasicBlock();
    BasicBlock* afterB = new BasicBlock();
    b.addSuccessor(trueB);
    b.addSuccessor(falseB);
    b.enterBasicBlock(trueB);
    visit(ifStmt->ifTrue);
    b.addSuccessor(afterB);
    b.enterBasicBlock(falseB);
    if (!ifStmt->ifFalse) {
        visit(ifStmt->ifFalse);
    }
    b.addSuccessor(afterB);
    b.enterBasicBlock(afterB);
    return false;
}

void CFGBuilder::Builder::add(const IR::StatOrDecl* item) {
    BUG_CHECK(curr, "");
    curr->components.push_back(item);
}

void CFGBuilder::Builder::addSuccessor(const BasicBlock* succ) {
    BUG_CHECK(curr, "");
    curr->succs.push_back(succ);
}

void CFGBuilder::Builder::enterBasicBlock(BasicBlock* bb) {
    BUG_CHECK(bb, "");
    curr = bb; 
}


} // namespace p4mlir