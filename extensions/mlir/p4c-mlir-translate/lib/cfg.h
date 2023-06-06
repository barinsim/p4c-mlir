#ifndef BACKENDS_MLIR_CFGBUILDER_H_
#define BACKENDS_MLIR_CFGBUILDER_H_

#include <vector>
#include <map>

#include "ir/ir.h"
#include "ir/visitor.h"

#include "lib/ordered_map.h"
#include "lib/ordered_set.h"


namespace p4mlir {


// Represents a lexical scope in a P4 program.
// The scope is represented using declarations created within the scope and its 'parent' scope.
class Scope
{
 public:
    Scope* parent = nullptr;
    ordered_set<const IR::IDeclaration*> decls;

 public:
    static Scope* create(Scope* parent_) { CHECK_NULL(parent_); return new Scope(parent_); }
    static Scope* create() { return new Scope(nullptr); }
    void add(const IR::IDeclaration* decl);

    // Checks if variable declared in 'decl' can be referred in this scope.
    // Recursively searches through the 'parent'.
    bool isVisible(const IR::IDeclaration* decl) const;

 private:
    Scope(Scope* parent_) : parent(parent_) {}
};

// TODO: fix const correctess of succs
struct BasicBlock
{
    BasicBlock(Scope& scope_) : scope(scope_) {}

    std::vector<const IR::Node*> components;
    std::vector<BasicBlock*> succs;

    // Stores visible declarations within this block
    Scope& scope;

    // Returns True/False successor of a block
    // terminated by IR::IfStatement.
    // Asserts that the block is terminated by IR::IfStatement
    const BasicBlock* getTrueSuccessor() const;
    const BasicBlock* getFalseSuccessor() const;

    static int nextId;
    int id = nextId++;
};

std::string toString(const BasicBlock* bb, int indent = 0);

// Control flow graph
class CFG
{
    BasicBlock* entry;

 public:
    CFG(BasicBlock* entry_) : entry(entry_) { CHECK_NULL(entry); }
    BasicBlock* getEntry() const { return entry; }
};

// Container of the control flow graphs.
// Associates CFGs with respective AST nodes
class CFGInfo
{
    ordered_map<const IR::Node*, CFG> data;

 public:
    // Add CFG 'cfg' and associate it with node 'node'
    void add(const IR::Node* node, CFG cfg) {
        CHECK_NULL(node);
        BUG_CHECK(!data.count(node), "CFG for node already exists");
        data.insert({node, cfg});
    }

    // Replace CFG that is associated with node 'node' by 'cfg'
    void replace(const IR::Node* node, CFG cfg) {
        CHECK_NULL(node);
        BUG_CHECK(contains(node), "Nothing to replace");
        data.insert({node, cfg});
    }

    // Return CFG that is associated with node 'node'
    CFG get(const IR::Node* node) const {
        CHECK_NULL(node);
        BUG_CHECK(data.count(node), "CFG for node does not exist");
        return data.at(node);
    }

    // Returns true if there is CFG that is associated with node 'node'
    bool contains(const IR::Node* node) const {
        CHECK_NULL(node);
        return data.count(node);
    }

    // Returns number of stored CFGs
    std::size_t size() const { return data.size(); }

    // Returns iterators over the stored data
    decltype(data.begin()) begin() { return data.begin(); }
    decltype(data.end()) end() { return data.end(); }
    decltype(data.cbegin()) begin() const { return data.cbegin(); }
    decltype(data.cend()) end() const { return data.cend(); }
};

// Creates control flow graph of all eligible P4 constructs in the program.
// Eligible P4 constructs:
//      Control block body (apply + out-of-apply local declarations)
//      Parser block body (out-of-apply local declarations)
//      Actions
//      States
class MakeCFGInfo : public Inspector
{
    // Output of this pass
    CFGInfo& cfgInfo;

    // Currently processed basic block
    BasicBlock* curr = nullptr;

 public:
    MakeCFGInfo(CFGInfo& cfgInfo_) : cfgInfo(cfgInfo_) {}

 private:
    BasicBlock* current() { return curr; }
    void addToCurrent(const IR::Node* item);
    void addSuccessorToCurrent(BasicBlock* succ);
    void enterBasicBlock(BasicBlock* bb);

    // Canonicalizes the CFGs:
    //  1. Block ending with return/exit has 0 successors
    //  2. There are no empty basic blocks (0 components)
    void end_apply(const IR::Node *) override;
    
    bool preorder(const IR::P4Action*) override;
    bool preorder(const IR::ParserState*) override;
    bool preorder(const IR::P4Control*) override;
    bool preorder(const IR::P4Parser*) override;
    bool preorder(const IR::IfStatement*) override;
    bool preorder(const IR::SwitchStatement*) override;
    bool preorder(const IR::Declaration_Variable*) override;
    bool preorder(const IR::ReturnStatement* ret) override { addToCurrent(ret); return true; }
    bool preorder(const IR::AssignmentStatement *assign) override {
        addToCurrent(assign);
        return true;
    }
    bool preorder(const IR::MethodCallStatement *call) override {
        addToCurrent(call);
        return true;
    }

    bool preorder(const IR::BlockStatement*) override {
        // Should 2 different scopes be in a single basic block?
        // Just ignore for now.
        return true;
    }
    bool preorder(const IR::ParameterList*) override { return false; }
    bool preorder(const IR::P4Table*) override { return false; }
    bool preorder(const IR::Annotations*) override { return false; }
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
                             ordered_set<const BasicBlock *> &visited,
                             std::ostream &os);
};

// Provides static methods for CFG inspecting/modifying
// TODO: Make these functions work with CFG instead of the entry BasicBlock
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
        ordered_set<BBType*> visited;
        postorder(entry, visited, func);
    }

    template <typename BBType, typename Func>
    static void preorder(BBType* entry, Func func) {
        CHECK_NULL(entry);
        ordered_set<BBType*> visited;
        preorder(entry, visited, func);
    }

    // Traverses the blocks from top to bottom as if they were ordered the same way as the P4
    // program from which they originate from
    template <typename BBType, typename Func>
    static void controlFlowTraversal(BBType* entry, Func func) {
        CHECK_NULL(entry);

        ordered_map<BBType*, int> indegree;
        ordered_map<BBType*, int> tag;
        computeIndegree(entry, indegree, tag);

        ordered_set<BBType*> visited;
        controlFlowTraversalImpl(entry, visited, func, indegree);
    }

    // TODO: This is not erase. More like follow-through
    // TODO: remove this
    template <typename Func>
    static BasicBlock* erase(BasicBlock* entry, Func shouldErase) {
        BasicBlock dummy(*Scope::create());
        dummy.succs.push_back(entry);
        forEachBlock(&dummy, [&dummy, &shouldErase](BasicBlock* bb) {
            if (shouldErase(bb) && bb != &dummy) {
                return;
            }
            std::vector<BasicBlock*> newSuccs;
            ordered_set<BasicBlock*> lookup;
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
    static void preorder(BBType* bb, ordered_set<BBType*>& visited, Func func) {
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
    static void postorder(BBType* bb, ordered_set<BBType*>& visited, Func func) {
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

    template <typename BBType, typename Func>
    static void controlFlowTraversalImpl(BBType *bb, ordered_set<BBType *> &visited, Func func,
                                         ordered_map<BBType*, int>& indegree) {
        if (visited.count(bb)) {
            return;
        }
        visited.insert(bb);
        func(bb);
        for (auto* succ : bb->succs) {
            --indegree.at(succ);
            BUG_CHECK(indegree.at(succ) >= 0, "Found more prdecessors than incoming degree");
            // Visit successor only if this is the last predecessor
            if (indegree.at(succ) == 0) {
                controlFlowTraversalImpl(succ, visited, func, indegree);
            }
        }
    }

    // Computes incoming degree for each block.
    // Ignores back edges (edge that goes into a node which has tag[node] == 1).
    // Cross and forward edges are ok
    template <typename BBType>
    static void computeIndegree(BBType *bb, ordered_map<BBType*, int>& indegree,
                                ordered_map<BBType*, int>& tag) {
        if (tag[bb] != 0) {
            return;
        }
        // Open
        tag[bb] = 1;
        for (auto* succ : bb->succs) {
            // Ignore back-edges
            if (tag[succ] == 1) {
                continue;
            }
            ++indegree[succ];
            computeIndegree(succ, indegree, tag);
        }
        // Close
        tag[bb] = 2;
    }
};


} // namespace p4mlir

#endif /* BACKENDS_MLIR_CFGBUILDER_H_ */