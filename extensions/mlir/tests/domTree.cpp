#include "gtest/gtest.h"

#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "../cfgBuilder.h"
#include "../domTree.h"


namespace p4mlir::tests {


// This is a simple way how to get 'BasicBlock' that contains 'stmt' statement.
// It relies on a unique string representation of the statement within the whole program.
BasicBlock* getByStmtString(BasicBlock* entry, const std::string& stmt) {
    auto bbs = p4mlir::CFGWalker::collect(entry, [&](auto* bb) {
        return std::any_of(bb->components.begin(), bb->components.end(), [&](auto* c) {
            return CFGPrinter::toString(c) == stmt;
        });
    });
    if (bbs.size() != 1) {
        throw std::domain_error("The searched statement must be unique and must exist");
    }
    return bbs.front();
}


class DomTree : public Test::P4CTest { };


TEST_F(DomTree, Test_action_with_control_flow) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            // bb1
            label = 1;
            if (hdr.f2 == hdr.f3 + 1) {
                // bb2
                label = 2;
                hdr.f1 = 3;
                if (hdr.f2 == hdr.f2 + 1) {
                    // bb3
                    label = 3;
                    hdr.f2 = 3;
                } else {
                    // bb4
                    label = 4;
                    hdr.f2 = 6;
                }
                // bb5
                label = 5;
                hdr.f6 = hdr.f1;
            }
            // bb6
            label = 6;
            test2.apply();
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), 1);
    auto* cfgFoo = all.begin()->second;
    ASSERT_TRUE(cfgFoo);

    p4mlir::BasicBlock* bb1 = getByStmtString(cfgFoo, "label = 1;");
    p4mlir::BasicBlock* bb2 = getByStmtString(cfgFoo, "label = 2;");
    p4mlir::BasicBlock* bb3 = getByStmtString(cfgFoo, "label = 3;");
    p4mlir::BasicBlock* bb4 = getByStmtString(cfgFoo, "label = 4;");
    p4mlir::BasicBlock* bb5 = getByStmtString(cfgFoo, "label = 5;");
    p4mlir::BasicBlock* bb6 = getByStmtString(cfgFoo, "label = 6;");

    p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgFoo);

    EXPECT_EQ(domTree->immediateDom(bb1), nullptr);
    EXPECT_EQ(domTree->immediateDom(bb2), bb1);
    EXPECT_EQ(domTree->immediateDom(bb3), bb2);
    EXPECT_EQ(domTree->immediateDom(bb4), bb2);
    EXPECT_EQ(domTree->immediateDom(bb5), bb2);
    EXPECT_EQ(domTree->immediateDom(bb6), bb1);

    // The order corresponds to the walk upwards from
    // the block to the entry block in a dominator tree
    using Blocks = std::vector<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->dominators(bb1), (Blocks{bb1}));
    EXPECT_EQ(domTree->dominators(bb2), (Blocks{bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb3), (Blocks{bb3, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb4), (Blocks{bb4, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb5), (Blocks{bb5, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb6), (Blocks{bb6, bb1}));
}


} // namespace p4mlir::tests
