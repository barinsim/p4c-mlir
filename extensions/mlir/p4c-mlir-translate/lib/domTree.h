#ifndef BACKENDS_MLIR_DOMTREE_H_
#define BACKENDS_MLIR_DOMTREE_H_

#include <vector>

#include "cfg.h"
#include "lib/ordered_map.h"
#include "lib/ordered_set.h"

namespace p4mlir {


class DomTree
{
 // Maps block to the index into 'data'.
 // Also corresponds to the postorder traversal of cfg.
 ordered_map<const BasicBlock*, int> mapping;
 ordered_map<int, const BasicBlock*> revMapping;

 // Tree data data[b] = 'parent of b'
 std::vector<int> data;

 // Tree data data[b] = 'children of b'
 std::vector<ordered_set<int>> revData;

 ordered_map<int, ordered_set<int>> domFrontiers;

 public:
    static DomTree* fromEntryBlock(const BasicBlock* entry) {
        CHECK_NULL(entry);
        return new DomTree(entry);
    }

    const BasicBlock* immediateDom(const BasicBlock* bb) const {
        CHECK_NULL(bb);
        int par = data.at(idx(bb));
        if (block(par) == bb) {
            return nullptr;
        }
        return block(par);
    }

    std::vector<const BasicBlock*> dominators(const BasicBlock* bb) const {
        CHECK_NULL(bb);
        std::vector<const BasicBlock*> res;
        res.push_back(bb);
        int node = idx(bb);
        while (data.at(node) != node) {
            node = data.at(node);
            res.push_back(block(node));
        }
        return res;
    }

    ordered_set<const BasicBlock*> domFrontier(const BasicBlock* bb) const {
        CHECK_NULL(bb);
        int node = idx(bb);
        BUG_CHECK(domFrontiers.count(node), "Dominance frontier info not found");
        ordered_set<const BasicBlock*> res;
        std::for_each(domFrontiers.at(node).begin(), domFrontiers.at(node).end(), [&](int n) {
            res.insert(block(n));
        });
        return res;
    }

    ordered_set<const BasicBlock*> children(const BasicBlock* bb) const {
        CHECK_NULL(bb);
        int node = idx(bb);
        BUG_CHECK(node >= 0 && node < (int)revData.size(), "Children info not found");
        ordered_set<const BasicBlock*> res;
        std::for_each(revData.at(node).begin(), revData.at(node).end(), [&](int n) {
            res.insert(block(n));
        });
        return res;
    }

 private:
    DomTree(const BasicBlock* entry);
    DomTree() = delete;

    const BasicBlock* block(int idx) const;
    int idx(const BasicBlock* bb) const;
};



} // namespace p4mlir


#endif /* BACKENDS_MLIR_DOMTREE_H_ */