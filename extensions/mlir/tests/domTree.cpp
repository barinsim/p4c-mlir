#include "gtest/gtest.h"


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
    auto* cfgFoo = all.front().second;
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
    EXPECT_EQ(domTree->dominators(bb1), std::vector(bb1));
    EXPECT_EQ(domTree->dominators(bb2), std::vector(bb2, bb1));
    EXPECT_EQ(domTree->dominators(bb3), std::vector(bb3, bb2, bb1));
    EXPECT_EQ(domTree->dominators(bb4), std::vector(bb4, bb2, bb1));
    EXPECT_EQ(domTree->dominators(bb5), std::vector(bb5, bb2, bb1));
    EXPECT_EQ(domTree->dominators(bb6), std::vector(bb6, bb1));
}