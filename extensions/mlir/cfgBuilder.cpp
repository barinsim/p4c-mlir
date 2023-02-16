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

bool CFGBuilder::preorder(const IR::P4Control* control) {
    BUG_CHECK(!b.callableToCFG.count(control), "");
    visit(control->controlLocals);
    BasicBlock* entryBlock = new BasicBlock();
    b.enterBasicBlock(entryBlock);
    b.callableToCFG.insert({control, entryBlock});
    visit(control->body);
    return false;
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
    if (ifStmt->ifFalse) {
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

// TODO: Refactor the whole CFG printing into its own CFG class
std::string toString(const BasicBlock* bb, int indent, bool followSuccessors,
                     std::unordered_set<const BasicBlock*>& visited) {
    auto bb_id = [](auto* bb) {
        return std::string("bb_") + std::to_string(bb->id);
    };

    auto terminatorToString = [](const IR::StatOrDecl* term) -> std::string {
        if (auto* ifstmt = term->to<IR::IfStatement>()) {
            std::stringstream ss;
            ss << "if ";
            ifstmt->condition->dbprint(ss);
            return ss.str();
        }
        if (auto* ret = term->to<IR::ReturnStatement>()) {
            return std::string("return");
        }
        std::stringstream ss;
        term->dbprint(ss);
        return ss.str();
    };

    visited.insert(bb);
    std::stringstream ss;

    ss << indent_t(indent) << bb_id(bb) << ":" << '\n';
    for (int i = 0; i < bb->components.size(); ++i) {
        auto* item = bb->components.at(i);
        if (i == bb->components.size() - 1) {
            ss << indent_t(indent + 1) << terminatorToString(item);
        } else {
            ss << indent_t(indent + 1);
            item->dbprint(ss);
            ss << '\n';
        }
    }

    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        ss << " " << bb_id(succ);
    });

    if (!followSuccessors) {
        return ss.str();
    }

    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        if (!visited.count(succ)) {
            ss << '\n' << '\n' << toString(succ, indent, true, visited);
        }
    });

    return ss.str();
}


} // namespace p4mlir