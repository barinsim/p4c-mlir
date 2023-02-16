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
    CHECK_NULL(curr);
    curr->components.push_back(item);
}

void CFGBuilder::Builder::addSuccessor(const BasicBlock* succ) {
    CHECK_NULL(curr);
    curr->succs.push_back(succ);
}

void CFGBuilder::Builder::enterBasicBlock(BasicBlock* bb) {
    CHECK_NULL(curr);
    curr = bb; 
}

std::string toString(const BasicBlock* bb, int indent) {
    CFGPrinter p;
    return p.toString(bb, indent);
}

std::string CFGPrinter::toString(const BasicBlock* entry, int indent) const {
    CHECK_NULL(entry);
    std::unordered_set<const BasicBlock*> visited;
    std::stringstream ss;
    toStringImpl(entry, indent, visited, ss);
    return ss.str();
}

void CFGPrinter::toStringImpl(const BasicBlock* bb, int indent,
                                     std::unordered_set<const BasicBlock*>& visited,
                                     std::ostream& os) const {
    visited.insert(bb);
    os << indent_t(indent) << getBlockIdentifier(bb);
    std::for_each(bb->components.begin(), bb->components.end(), [&](auto* comp) {
        os << '\n' << indent_t(indent + 1) << toString(comp);
    });
    if (!bb->succs.empty()) {
        os << '\n' << indent_t(indent + 1) << "successors:";
        std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
            os << " " << getBlockIdentifier(succ);
        });
    }
    bool willContinue = std::any_of(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        return !visited.count(succ);
    });
    if (willContinue) {
        os << '\n' << '\n';
    }
    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        if (!visited.count(succ)) {
            toStringImpl(succ, indent, visited, os);
        }
    });
}

std::string CFGPrinter::toString(const IR::Node* node) const {
    CHECK_NULL(node);
    std::stringstream ss;
    if (auto* ifstmt = node->to<IR::IfStatement>()) {
        ss << "if (";
        ifstmt->condition->dbprint(ss);
        ss << ")";
        return ss.str();
    }
    node->dbprint(ss);
    return ss.str();
}

std::string CFGPrinter::getBlockIdentifier(const BasicBlock* bb) const {
    CHECK_NULL(bb);
    return std::string("bb^") + std::to_string(bb->id);
}


} // namespace p4mlir