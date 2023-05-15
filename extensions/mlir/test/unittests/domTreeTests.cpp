#include "gtest/gtest.h"

#include <unordered_set>

#include "frontends/common/parseInput.h"
#include "test/gtest/helpers.h"

#include "cfg.h"
#include "domTree.h"
#include "common.h"


namespace p4mlir::tests {


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
            test.apply();
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    p4mlir::CFGInfo cfgInfo;
    auto b = new p4mlir::MakeCFGInfo(cfgInfo);
    pgm->apply(*b);

    ASSERT_EQ(cfgInfo.size(), (std::size_t)1);
    p4mlir::CFG cfgFoo = cfgInfo.begin()->second;

    p4mlir::BasicBlock* bb1 = getByStmtString(cfgFoo, "label = 1;");
    p4mlir::BasicBlock* bb2 = getByStmtString(cfgFoo, "label = 2;");
    p4mlir::BasicBlock* bb3 = getByStmtString(cfgFoo, "label = 3;");
    p4mlir::BasicBlock* bb4 = getByStmtString(cfgFoo, "label = 4;");
    p4mlir::BasicBlock* bb5 = getByStmtString(cfgFoo, "label = 5;");
    p4mlir::BasicBlock* bb6 = getByStmtString(cfgFoo, "label = 6;");

    p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgFoo.getEntry());

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

    using UnorderedBlocks = std::unordered_set<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->domFrontier(bb1), (UnorderedBlocks{}));
    EXPECT_EQ(domTree->domFrontier(bb2), (UnorderedBlocks{bb6}));
    EXPECT_EQ(domTree->domFrontier(bb3), (UnorderedBlocks{bb5}));
    EXPECT_EQ(domTree->domFrontier(bb4), (UnorderedBlocks{bb5}));
    EXPECT_EQ(domTree->domFrontier(bb5), (UnorderedBlocks{bb6}));
    EXPECT_EQ(domTree->domFrontier(bb6), (UnorderedBlocks{}));
}

TEST_F(DomTree, Test_action_with_complex_control_flow) {
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
                    if (hdr.f3) {
                        // bb4
                        label = 4;
                    } else {
                        // bb5
                        label = 5;
                    }
                } else {
                    // bb6
                    if (label == 6) {
                        // bb7
                        label = 7;
                    }
                }
            }
            // bb8
            label = 8;
            test.apply();
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    p4mlir::CFGInfo cfgInfo;
    auto b = new p4mlir::MakeCFGInfo(cfgInfo);
    pgm->apply(*b);

    ASSERT_EQ(cfgInfo.size(), (std::size_t)1);
    p4mlir::CFG cfgFoo = cfgInfo.begin()->second;

    p4mlir::BasicBlock* bb1 = getByStmtString(cfgFoo, "label = 1;");
    p4mlir::BasicBlock* bb2 = getByStmtString(cfgFoo, "label = 2;");
    p4mlir::BasicBlock* bb3 = getByStmtString(cfgFoo, "label = 3;");
    p4mlir::BasicBlock* bb4 = getByStmtString(cfgFoo, "label = 4;");
    p4mlir::BasicBlock* bb5 = getByStmtString(cfgFoo, "label = 5;");
    p4mlir::BasicBlock* bb6 = getByStmtString(cfgFoo, "if (label == 6)");
    p4mlir::BasicBlock* bb7 = getByStmtString(cfgFoo, "label = 7;");
    p4mlir::BasicBlock* bb8 = getByStmtString(cfgFoo, "label = 8;");

    p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgFoo.getEntry());

    EXPECT_EQ(domTree->immediateDom(bb1), nullptr);
    EXPECT_EQ(domTree->immediateDom(bb2), bb1);
    EXPECT_EQ(domTree->immediateDom(bb3), bb2);
    EXPECT_EQ(domTree->immediateDom(bb4), bb3);
    EXPECT_EQ(domTree->immediateDom(bb5), bb3);
    EXPECT_EQ(domTree->immediateDom(bb6), bb2);
    EXPECT_EQ(domTree->immediateDom(bb7), bb6);
    EXPECT_EQ(domTree->immediateDom(bb8), bb1);

    using Blocks = std::vector<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->dominators(bb1), (Blocks{bb1}));
    EXPECT_EQ(domTree->dominators(bb2), (Blocks{bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb3), (Blocks{bb3, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb4), (Blocks{bb4, bb3, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb5), (Blocks{bb5, bb3, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb6), (Blocks{bb6, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb7), (Blocks{bb7, bb6, bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb8), (Blocks{bb8, bb1}));

    using UnorderedBlocks = std::unordered_set<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->domFrontier(bb1), (UnorderedBlocks{}));
    EXPECT_EQ(domTree->domFrontier(bb2), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb3), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb4), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb5), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb6), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb7), (UnorderedBlocks{bb8}));
    EXPECT_EQ(domTree->domFrontier(bb8), (UnorderedBlocks{}));
}

TEST_F(DomTree, Test_fall_through_switch_statement) {
    std::string src = P4_SOURCE(R"(
        struct Parsed_packet {}
        control Pipe<H>(inout H headers);
        package Pipeline<H>(Pipe<H> p);
        control TopPipe(inout Parsed_packet hdr) {
            action foo1() {}
            action foo2() {}
            action foo3() {}
            table test_table {
                key = { hdr.f1: exact; }
                actions = { foo1; foo2; foo3; }
                size = 1024;
            }
            apply {
                // bb1
                label = 1;
                switch (test_table.apply().action_run) {
                    foo1 : {
                        // bb2
                        label = 2;
                    }
                    foo2 :
                    foo3 : {
                        // bb3
                        label = 3;
                        if (hdr.f1 == 10) {
                            // bb4
                            label = 4;
                            return;
                        }
                    }
                    foo4 :
                    default: {
                        // bb5
                        if (label == 5) {
                            // bb6
                            label = 6;
                        }
                    }
                }
                // bb7
                label = 7;
                if (hdr.f1 == 3) {
                    // bb8
                    label = 8;
                }
                // bb9
                label = 9;
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    p4mlir::CFGInfo cfgInfo;
    auto b = new p4mlir::MakeCFGInfo(cfgInfo);
    pgm->apply(*b);

    ASSERT_EQ(cfgInfo.size(), (std::size_t)4);
    auto cfgApply = getByName(cfgInfo, "TopPipe");

    p4mlir::BasicBlock* bb1 = getByStmtString(cfgApply, "label = 1;");
    p4mlir::BasicBlock* bb2 = getByStmtString(cfgApply, "label = 2;");
    p4mlir::BasicBlock* bb3 = getByStmtString(cfgApply, "label = 3;");
    p4mlir::BasicBlock* bb4 = getByStmtString(cfgApply, "label = 4;");
    p4mlir::BasicBlock* bb5 = getByStmtString(cfgApply, "if (label == 5)");
    p4mlir::BasicBlock* bb6 = getByStmtString(cfgApply, "label = 6;");
    p4mlir::BasicBlock* bb7 = getByStmtString(cfgApply, "label = 7;");
    p4mlir::BasicBlock* bb8 = getByStmtString(cfgApply, "label = 8;");
    p4mlir::BasicBlock* bb9 = getByStmtString(cfgApply, "label = 9;");

    p4mlir::DomTree* domTree = p4mlir::DomTree::fromEntryBlock(cfgApply.getEntry());

    EXPECT_EQ(domTree->immediateDom(bb1), nullptr);
    EXPECT_EQ(domTree->immediateDom(bb2), bb1);
    EXPECT_EQ(domTree->immediateDom(bb3), bb1);
    EXPECT_EQ(domTree->immediateDom(bb4), bb3);
    EXPECT_EQ(domTree->immediateDom(bb5), bb1);
    EXPECT_EQ(domTree->immediateDom(bb6), bb5);
    EXPECT_EQ(domTree->immediateDom(bb7), bb1);
    EXPECT_EQ(domTree->immediateDom(bb8), bb7);
    EXPECT_EQ(domTree->immediateDom(bb9), bb7);

    using Blocks = std::vector<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->dominators(bb1), (Blocks{bb1}));
    EXPECT_EQ(domTree->dominators(bb2), (Blocks{bb2, bb1}));
    EXPECT_EQ(domTree->dominators(bb3), (Blocks{bb3, bb1}));
    EXPECT_EQ(domTree->dominators(bb4), (Blocks{bb4, bb3, bb1}));
    EXPECT_EQ(domTree->dominators(bb5), (Blocks{bb5, bb1}));
    EXPECT_EQ(domTree->dominators(bb6), (Blocks{bb6, bb5, bb1}));
    EXPECT_EQ(domTree->dominators(bb7), (Blocks{bb7, bb1}));
    EXPECT_EQ(domTree->dominators(bb8), (Blocks{bb8, bb7, bb1}));
    EXPECT_EQ(domTree->dominators(bb9), (Blocks{bb9, bb7, bb1}));

    using UnorderedBlocks = std::unordered_set<const p4mlir::BasicBlock*>;
    EXPECT_EQ(domTree->domFrontier(bb1), (UnorderedBlocks{}));
    EXPECT_EQ(domTree->domFrontier(bb2), (UnorderedBlocks{bb7}));
    EXPECT_EQ(domTree->domFrontier(bb3), (UnorderedBlocks{bb7}));
    EXPECT_EQ(domTree->domFrontier(bb4), (UnorderedBlocks{}));
    EXPECT_EQ(domTree->domFrontier(bb5), (UnorderedBlocks{bb7}));
    EXPECT_EQ(domTree->domFrontier(bb6), (UnorderedBlocks{bb7}));
    EXPECT_EQ(domTree->domFrontier(bb7), (UnorderedBlocks{}));
    EXPECT_EQ(domTree->domFrontier(bb8), (UnorderedBlocks{bb9}));
}


} // namespace p4mlir::tests
