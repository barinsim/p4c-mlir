#include "domTree.h"


namespace p4mlir {


DomTree::DomTree(const BasicBlock* entry) {
    CHECK_NULL(entry);

    // Assigns number to each block corresponding to its postorder traversal index.
    // Once this mapping is established the whole data structure uses only these indices.
    // Methods 'block()' and 'idx()' can be used to convert index back to block and vice versa.
    auto createMapping = [](const BasicBlock* e) {
        int num = 0;
        std::unordered_map<const BasicBlock*, int> mp;
        CFGWalker::postorder(e, [&num, &mp](const BasicBlock* bb) {
            BUG_CHECK(!mp.count(bb), "Each block must be visited once");
            mp[bb] = num;
            ++num;
        });
        return mp;
    };

    auto collectPredecessors = [&](const BasicBlock* e) {
        BUG_CHECK(!mapping.empty(), "Mapping must be already established at this point");
        std::unordered_map<int, std::vector<int>> preds;
        CFGWalker::forEachBlock(e, [&](const BasicBlock* bb) {
            if (!preds.count(idx(bb))) {
                preds.insert({idx(bb), {}});
            }
            for (auto* succ : bb->succs) {
                preds[idx(succ)].push_back(idx(bb));
            }
        });
        return preds;
    };

    // Walks up the 'par' tree, returning the first common predecessor.
    // It uses the fact that in a dominator tree parent index is higher
    // than child index.
    auto commonPred = [](int a, int b, const std::vector<int> &par) {
        BUG_CHECK(a >= 0 && b >= 0 && a < (int)par.size() && b < (int)par.size(),
                  "Nodes must exist in the tree");
        while (a != b) {
            if (a < b) {
                a = par[a];
            } else {
                b = par[b];
            }
        }
        return a;
    };

    // Creates the dominator tree
    auto createTree = [&]() {
        int nodes = mapping.size();
        int entry = nodes - 1;
        const auto preds = collectPredecessors(block(entry));
        std::vector<int> par(nodes, -1);
        par[entry] = entry;
        bool changed = true;
        while (changed) {
            changed = false;
            // Traverse nodes in a reverse postorder (except the start node)
            for (int node = nodes - 2; node >= 0; --node) {
                BUG_CHECK(preds.count(node), "Predecessors info not found");
                int immDom = -1;
                for (int pred : preds.at(node)) {
                    if (par[pred] == -1) {
                        continue;
                    }
                    if (immDom == -1) {
                        immDom = pred;
                        continue;
                    }
                    immDom = commonPred(immDom, pred, par);
                }
                BUG_CHECK(immDom != -1, "Could not find dominators for a block");
                if (immDom != par[node]) {
                    par[node] = immDom;
                    changed = true;
                }
            }
        }
        return par;
    };

    // Creates 'node -> {children}' tree from 'node -> parent' tree
    auto createRevTree = [&](const std::vector<int>& par) {
        std::vector<std::unordered_set<int>> res(par.size());
        for (int node = 0; node < (int)par.size(); ++node) {
            auto status = res[par[node]].insert(node);
            BUG_CHECK(status.second, "Node has multiple identical children");
        }
        return res;
    };

    // For each node creates a set of dominance frontier nodes,
    // Using dominator tree 'par'
    auto createDominanceFrontierSets = [&](const std::vector<int>& par) {
        int nodes = mapping.size();
        int entry = nodes - 1;
        std::unordered_map<int, std::unordered_set<int>> res;
        const auto preds = collectPredecessors(block(entry));
        for (int node = 0; node < nodes; ++node) {
            BUG_CHECK(preds.count(node), "Predecessors info not found");
            if (!res.count(node)) {
                res.insert({node, {}});
            }
            if (preds.at(node).size() < 2) {
                continue;
            }
            for (int pred : preds.at(node)) {
                int ptr = pred;
                while (ptr != par.at(node)) {
                    res[ptr].insert(node);
                    ptr = par.at(ptr);
                }
            }
        }
        return res;
    };

    mapping = createMapping(entry);
    for (auto& [bb, idx] : mapping) {
        BUG_CHECK(!revMapping.count(idx), "The mapping must be one-to-one");
        revMapping[idx] = bb;
    }
    BUG_CHECK(block(mapping.size() - 1) == entry,
                "Entry block should map to the last index in a postorder traversal");

    data = createTree();
    revData = createRevTree(data);
    domFrontiers = createDominanceFrontierSets(data);
}

const BasicBlock* DomTree::block(int idx) const {
    BUG_CHECK(revMapping.count(idx), "Mapping for ", idx, " does not exist");
    return revMapping.at(idx);
}

int DomTree::idx(const BasicBlock* bb) const {
    BUG_CHECK(mapping.count(bb),
                "Mapping for ", CFGPrinter::makeBlockIdentifier(bb), " does not exist");
    return mapping.at(bb);
}


} // namespace p4mlir
