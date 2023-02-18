#include "cfgBuilder.h"

#include "lib/indent.h"


namespace p4mlir {


int BasicBlock::nextId = 0;

void CFGBuilder::end_apply(const IR::Node*) {
    auto isFinal = [](const BasicBlock* bb) {
        return bb->succs.empty();
    };

    auto isExitBlock = [](const BasicBlock* bb) {
        if (bb->components.empty()) {
            return false;
        }
        auto* last = bb->components.back();
        // TODO: Add other terminators
        return last->is<IR::ReturnStatement>();
    };

    auto isRedundant = [](const BasicBlock* bb) {
        return bb->components.empty();
    };

    for (auto& [decl, entry] : b.callableToCFG) {
        auto finalBlocks = CFGWalker::collect(entry, isFinal);
        int modifiedBlocks = 0;
        for (auto* bb : finalBlocks) {
            // TODO: Add other terminators
            if (bb->components.empty() || !bb->components.back()->is<IR::ReturnStatement>()) {
                bb->components.push_back(new IR::ReturnStatement(nullptr));
            }
        }
        BUG_CHECK(modifiedBlocks <= 1,
                  "There should be at most 1 final block with missing exit statement.");
    }

    for (auto& [decl, entry] : b.callableToCFG) {
        auto blocks = CFGWalker::collect(entry, isExitBlock);
        std::for_each(blocks.begin(), blocks.end(), [](auto* bb) { bb->succs.clear(); });
    }

    for (auto& [decl, entry] : b.callableToCFG) {
        entry = CFGWalker::erase(entry, isRedundant);
    }
}

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
    CHECK_NULL(curr, item);
    LOG3("CFGBuilder::add " << DBPrint::Brief << item);
    curr->components.push_back(item);
}

void CFGBuilder::Builder::addSuccessor(BasicBlock* succ) {
    CHECK_NULL(curr, succ);
    curr->succs.push_back(succ);
}

void CFGBuilder::Builder::enterBasicBlock(BasicBlock* bb) {
    CHECK_NULL(bb);
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
    std::string str = ss.str();
    int pos = str.find_last_of('\n');
    str.erase(pos);
    return str;
}

void CFGPrinter::toStringImpl(const BasicBlock* bb, int indent,
                              std::unordered_set<const BasicBlock*>& visited,
                              std::ostream& os) const {
    visited.insert(bb);
    os << indent_t(indent) << makeBlockIdentifier(bb) << '\n';
    std::for_each(bb->components.begin(), bb->components.end(), [&](auto* comp) {
        os << indent_t(indent + 1) << toString(comp) << '\n';
    });
    os << indent_t(indent + 1) << "successors:";
    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        os << " " << makeBlockIdentifier(succ);
    });
    os << "\n\n";
    std::for_each(bb->succs.begin(), bb->succs.end(), [&](auto* succ) {
        if (!visited.count(succ)) {
            toStringImpl(succ, indent, visited, os);
        }
    });
}

std::string CFGPrinter::toString(const IR::Node* node) const {
    CHECK_NULL(node);
    if (auto* ifstmt = node->to<IR::IfStatement>()) {
        std::stringstream ss;
        ifstmt->condition->dbprint(ss);
        std::string cond = ss.str();
        if (!cond.empty() && cond.back() == ';') {
            cond.pop_back();
        }
        return std::string("if (") + cond + ")";
    }
    std::stringstream ss;
    node->dbprint(ss);
    return ss.str();
}

std::string CFGPrinter::makeBlockIdentifier(const BasicBlock* bb) {
    CHECK_NULL(bb);
    return std::string("bb^") + std::to_string(bb->id);
}


} // namespace p4mlir