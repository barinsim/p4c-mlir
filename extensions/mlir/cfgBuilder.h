#ifndef BACKENDS_MLIR_CFGBUILDER_H_
#define BACKENDS_MLIR_CFGBUILDER_H_

#include <vector>
#include <map>

#include "ir/ir.h"
#include "ir/visitor.h"

namespace p4mlir {


// TODO: fix const correctes of succs
struct BasicBlock {
    std::vector<const IR::StatOrDecl*> components;
    std::vector<BasicBlock*> succs;
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
        BasicBlock* current() { return curr; }
        void add(const IR::StatOrDecl* item);
        void addSuccessor(BasicBlock* succ);
        void enterBasicBlock(BasicBlock* bb);
    };

    Builder b;

    Visitor::profile_t init_apply(const IR::Node *node) override {
        b = Builder();
        return Inspector::init_apply(node);
    }

    // Canonicalizes the CFGs:
    //  1. The last statement of a callable is a return statement
    //  2. Block ending with return/exit has 0 successors
    //  3. There are no empty basic blocks (0 components)
    void end_apply(const IR::Node *) override;
    
    bool preorder(const IR::P4Action* action) override;
    bool preorder(const IR::P4Control* control) override;
    bool preorder(const IR::IfStatement* ifStmt) override;
    bool preorder(const IR::SwitchStatement* switchStmt) override;
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


// TODO: reformat with CFGWalker
class CFGPrinter
{
 public:
    static std::string toString(const BasicBlock* entry, int indent = 0);
    static std::string toString(const IR::Node* node);

public:
    static std::string makeBlockIdentifier(const BasicBlock* bb);

 private:
    static void toStringImpl(const BasicBlock *bb, int indent,
                             std::unordered_set<const BasicBlock *> &visited,
                             std::ostream &os);
};


// Provides static methods for CFG inspecting/modifying
class CFGWalker
{
 public:
    template <typename BBType, typename Func>
    static std::vector<BBType*> collect(BBType* entry, Func shouldCollect) {
        std::vector<BBType*> res;
        forEachBlock(entry, [&res, &shouldCollect](BBType* bb) {
            if (shouldCollect(bb)) {
                res.push_back(bb);
            }
        });
        return res;
    }

    template <typename BBType, typename Func>
    static void forEachBlock(BBType* entry, Func func) {
        preorder(entry, func);
    }

    template <typename BBType, typename Func>
    static void postorder(BBType* entry, Func func) {
        CHECK_NULL(entry);
        std::unordered_set<BBType*> visited;
        postorder(entry, visited, func);
    }

    template <typename BBType, typename Func>
    static void preorder(BBType* entry, Func func) {
        CHECK_NULL(entry);
        std::unordered_set<BBType*> visited;
        preorder(entry, visited, func);
    }

    // TODO: This is not erase. More like follow-through
    template <typename Func>
    static BasicBlock* erase(BasicBlock* entry, Func shouldErase) {
        BasicBlock dummy;
        dummy.succs.push_back(entry);
        forEachBlock(&dummy, [&dummy, &shouldErase](BasicBlock* bb) {
            if (shouldErase(bb) && bb != &dummy) {
                return;
            }
            std::vector<BasicBlock*> newSuccs;
            std::unordered_set<BasicBlock*> lookup;
            for (auto* succ : bb->succs) {
                // We need to get over all of which should
                // be erased to the first valid one
                BasicBlock* ptr = succ;
                while (ptr && shouldErase(ptr)) {
                    BUG_CHECK(ptr->succs.size() <= 1,
                              "Vertices with more than 1 succesor cannot be removed.");
                    if (ptr->succs.empty()) {
                        ptr = nullptr;
                        continue;
                    }
                    ptr = ptr->succs.front();
                }
                if (ptr && !lookup.count(ptr)) {
                    newSuccs.push_back(ptr);
                    lookup.insert(ptr);
                }
            }
            bb->succs = newSuccs;
        });
        return dummy.succs.empty() ? nullptr : dummy.succs.front();
    }

 private:
    template <typename BBType, typename Func>
    static void preorder(BBType* bb, std::unordered_set<BBType*>& visited, Func func) {
        CHECK_NULL(bb);
        if (visited.count(bb)) {
            return;
        }
        visited.insert(bb);
        func(bb);
        for (auto* succ : bb->succs) {
            preorder((BBType*)succ, visited, func);
        }
    }

    template <typename BBType, typename Func>
    static void postorder(BBType* bb, std::unordered_set<BBType*>& visited, Func func) {
        CHECK_NULL(bb);
        if (visited.count(bb)) {
            return;
        }
        visited.insert(bb);
        for (auto* succ : bb->succs) {
            postorder((BBType*)succ, visited, func);
        }
        func(bb);
    }
};


} // namespace p4mlir

#endif /* BACKENDS_MLIR_CFGBUILDER_H_ */