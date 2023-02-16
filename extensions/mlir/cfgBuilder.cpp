#include "cfgBuilder.h"

#include "lib/indent.h"


namespace p4mlir {


int BasicBlock::nextId = 0;

bool CFGBuilder::preorder(const IR::P4Action* action) {
        BUG_CHECK(!b.callableToCFG.count(action), "");
        BasicBlock* entryBlock = new BasicBlock();
        b.enterBasicBlock(entryBlock);
        b.callableToCFG.insert({action, entryBlock});
        return true;
}

bool CFGBuilder::preorder(const IR::StatOrDecl* statOrDecl) {
    if (!findContext<IR::P4Action>()) {
        return false;
    }
    b.add(statOrDecl);
    return true;
}

bool CFGBuilder::preorder(const IR::IfStatement* ifStmt) {
    if (!findContext<IR::P4Action>()) {
        return false;
    }
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

// TODO: Refactor the whole CFG printing into if own CFG class
std::string toString(const BasicBlock* bb, int indent, bool followSuccessors,
                     std::unordered_set<const BasicBlock*> visited) {
    auto bb_id = [](auto* bb) {
        return std::string("bb_") + std::to_string(bb->id);
    };

    visited.insert(bb);
    std::stringstream ss;

    ss << indent_t(indent) << bb_id(bb) << ":" << '\n';
    std::for_each(bb->components.begin(), bb->components.end(), [&](auto* item) {
        ss << indent_t(indent + 1);
        item->dbprint(ss);
        ss << '\n';
    });

    ss << indent_t(indent + 1) << "successors:";
    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        ss << " " << bb_id(succ);
    });

    if (!followSuccessors) {
        return ss.str();
    }

    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        if (!visited.count(succ)) {
            ss << '\n' << '\n' << toString(succ, indent, true);
        }
    });

    return ss.str();
}


} // namespace p4mlir