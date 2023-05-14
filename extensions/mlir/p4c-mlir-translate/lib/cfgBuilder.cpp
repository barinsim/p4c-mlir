#include "cfgBuilder.h"

#include <boost/algorithm/string.hpp>

#include "lib/indent.h"

#include "frontends/p4/toP4/toP4.h"


namespace p4mlir {


int BasicBlock::nextId = 0;

void Scope::add(const IR::IDeclaration* decl) {
    CHECK_NULL(decl);
    BUG_CHECK(!decls.count(decl), "Declaration is already present in this scope");
    decls.insert(decl);
}

bool Scope::isVisible(const IR::IDeclaration* decl) const {
    CHECK_NULL(decl);
    if (decls.count(decl)) {
        return true;
    }
    if (!parent) {
        return false;
    }
    return parent->isVisible(decl);
}

const BasicBlock* BasicBlock::getTrueSuccessor() const {
    BUG_CHECK(!components.empty(), "BasicBlock is empty");
    BUG_CHECK(components.back()->is<IR::IfStatement>(),
              "BasicBlock is not terminated by IR::IfStatement");
    return succs.at(0);
}

const BasicBlock* BasicBlock::getFalseSuccessor() const {
    BUG_CHECK(!components.empty(), "BasicBlock is empty");
    BUG_CHECK(components.back()->is<IR::IfStatement>(),
              "BasicBlock is not terminated by IR::IfStatement");

    // There are some wierd cases where if stmt can have 1 successor,
    // in that case False target is a True target.
    // TODO: it stems from the canonicalization of the CFG, remove it
    if (succs.size() == 1) {
        return succs.at(0);
    }
    return succs.at(1);
}

void MakeCFGInfo::end_apply(const IR::Node*) {
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

    for (auto& [node, cfg] : cfgInfo) {
        auto finalBlocks = CFGWalker::collect(cfg.getEntry(), isFinal);
        int modifiedBlocks = 0;
        for (auto* bb : finalBlocks) {
            // TODO: Add other terminators
            if (bb->components.empty() || !bb->components.back()->is<IR::ReturnStatement>()) {
                bb->components.push_back(new IR::ReturnStatement(nullptr));
                ++modifiedBlocks;
            }
        }
        BUG_CHECK(modifiedBlocks <= 1,
                  "There should be at most 1 final block with missing exit statement.");
    }

    for (auto& [node, cfg] : cfgInfo) {
        auto blocks = CFGWalker::collect(cfg.getEntry(), isExitBlock);
        std::for_each(blocks.begin(), blocks.end(), [](auto* bb) { bb->succs.clear(); });
    }

    std::vector<std::pair<const IR::Node*, CFG>> toReplace;
    for (auto& [node, cfg] : cfgInfo) {
        auto* newEntry = CFGWalker::erase(cfg.getEntry(), isRedundant);
        toReplace.push_back({node, newEntry});
    }
    std::for_each(toReplace.begin(), toReplace.end(), [&](auto& kv) {
        cfgInfo.replace(kv.first, kv.second);
    });
}

bool MakeCFGInfo::preorder(const IR::P4Action* action) {
    BUG_CHECK(!cfgInfo.contains(action), "Action already visited");
    BasicBlock* entryBlock = new BasicBlock(*Scope::create());
    enterBasicBlock(entryBlock);
    cfgInfo.add(action, entryBlock);
    return true;
}

bool MakeCFGInfo::preorder(const IR::P4Control* control) {
    BUG_CHECK(!cfgInfo.contains(control), "Control visited twice");

    // Create CFG for actions
    visit(control->controlLocals);

    // Create CFG for the body of the block (out-of-apply declarations + apply)
    auto* entryBlock = new BasicBlock(*Scope::create());
    enterBasicBlock(entryBlock);
    cfgInfo.add(control, entryBlock);

    // Add the out-of-apply variable locals of the control block into the same CFG as the apply
    // method
    std::for_each(control->controlLocals.begin(), control->controlLocals.end(),
                  [&](const IR::Declaration *decl) {
                      if (decl->is<IR::Declaration_Variable>()) {
                          addToCurrent(decl);
                      }
                  });

    visit(control->body);
    return false;
}

bool MakeCFGInfo::preorder(const IR::IfStatement* ifStmt) {
    addToCurrent(ifStmt);
    Scope& scope = current()->scope;
    BasicBlock* trueB = new BasicBlock(*Scope::create(&scope));
    BasicBlock* falseB = new BasicBlock(*Scope::create(&scope));
    BasicBlock* afterB = new BasicBlock(scope);
    addSuccessorToCurrent(trueB);
    addSuccessorToCurrent(falseB);
    enterBasicBlock(trueB);
    visit(ifStmt->ifTrue);
    addSuccessorToCurrent(afterB);
    enterBasicBlock(falseB);
    if (ifStmt->ifFalse) {
        visit(ifStmt->ifFalse);
    }
    addSuccessorToCurrent(afterB);
    enterBasicBlock(afterB);
    return false;
}

bool MakeCFGInfo::preorder(const IR::SwitchStatement* switchStmt) {
    addToCurrent(switchStmt);
    BasicBlock* beforeB = current();
    Scope& scope = beforeB->scope;

    auto createBlockForEachCase = [&](auto& cases) {
        std::vector<BasicBlock*> res;
        std::for_each(cases.begin(), cases.end(), [&](auto*) {
            res.push_back(new BasicBlock(*Scope::create(&scope)));
        });
        return res;
    };

    auto isFallthrough = [](auto* c) {
        return !c->statement;
    };

    auto hasDefault = [](const IR::Vector<IR::SwitchCase>& cases) {
        return !cases.empty() && cases.back()->label->is<IR::DefaultExpression>();
    };

    auto& cases = switchStmt->cases;
    std::vector<BasicBlock*> blocks = createBlockForEachCase(cases);
    // Add the 'after' block
    BasicBlock* afterB = new BasicBlock(scope);
    blocks.push_back(afterB);
    for (std::size_t i = 0; i < cases.size(); ++i) {
        enterBasicBlock(beforeB);
        addSuccessorToCurrent(blocks[i]);
        enterBasicBlock(blocks[i]);
        if (isFallthrough(cases[i])) {
            addSuccessorToCurrent(blocks[i + 1]);
            continue;
        }
        visit(cases[i]);
        addSuccessorToCurrent(afterB);
    }
    enterBasicBlock(beforeB);
    if (!hasDefault(cases)) {
        addSuccessorToCurrent(afterB);
    }
    enterBasicBlock(afterB);
    return false;
}

bool MakeCFGInfo::preorder(const IR::Declaration_Variable* decl) {
    if (!findContext<IR::BlockStatement>()) {
        return true;
    }
    addToCurrent(decl);
    current()->scope.add(decl);
    return true;
}

void MakeCFGInfo::addToCurrent(const IR::StatOrDecl* item) {
    CHECK_NULL(curr, item);
    LOG3("CFGBuilder::add " << DBPrint::Brief << item);
    curr->components.push_back(item);
}

void MakeCFGInfo::addSuccessorToCurrent(BasicBlock* succ) {
    CHECK_NULL(curr, succ);
    curr->succs.push_back(succ);
}

void MakeCFGInfo::enterBasicBlock(BasicBlock* bb) {
    CHECK_NULL(bb);
    curr = bb;
}

std::string toString(const BasicBlock* bb, int indent) {
    return CFGPrinter::toString(bb, indent);
}

std::string CFGPrinter::toString(const BasicBlock* entry, int indent) {
    CHECK_NULL(entry);
    ordered_set<const BasicBlock*> visited;
    std::stringstream ss;
    toStringImpl(entry, indent, visited, ss);
    std::string str = ss.str();
    int pos = str.find_last_of('\n');
    str.erase(pos);
    return str;
}

void CFGPrinter::toStringImpl(const BasicBlock* bb, int indent,
                              ordered_set<const BasicBlock*>& visited,
                              std::ostream& os) {
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

std::string CFGPrinter::toString(const IR::Node* node) {
    CHECK_NULL(node);

    // Creates a string for all cases which are run on the same action
    auto groupToString = [&](const std::vector<const IR::SwitchCase*>& group) {
        std::string res;
        res += "[";
        for (auto* c : group) {
            res += toString(c->label);
            if (res.back() == ';') {
                res.pop_back();
            }
            res += " ";
        }
        boost::algorithm::trim(res);
        res += "]";
        return res;
    };

    // switch (table_t1 [foo1][foo2 foo3 foo4][foo5][default])
    auto switchToString = [&](const IR::SwitchStatement* switchStmt) {
        std::string table = toString(switchStmt->expression);
        auto pos = table.find('.');
        if (pos != std::string::npos) {
            table = table.substr(0, pos);
        }
        std::vector<std::vector<const IR::SwitchCase*>> groups;
        groups.emplace_back();
        for (auto* c : switchStmt->cases) {
            CHECK_NULL(c);
            groups.back().push_back(c);
            if (c->statement) {
                groups.emplace_back();
            }
        }
        if (groups.back().empty()) {
            groups.pop_back();
        }
        std::string labels;
        for (auto& g : groups) {
            labels += groupToString(g);
        }
        return std::string("switch (") + table + " " + labels + ")";
    };

    if (auto* ifStmt = node->to<IR::IfStatement>()) {
        std::string cond = toString(ifStmt->condition);
        return std::string("if (") + cond + ")";
    }
    if (auto* switchStmt = node->to<IR::SwitchStatement>()) {
        return switchToString(switchStmt);
    }
    return P4::toP4(node);
}

std::string CFGPrinter::makeBlockIdentifier(const BasicBlock* bb) {
    CHECK_NULL(bb);
    return std::string("bb^") + std::to_string(bb->id);
}


} // namespace p4mlir