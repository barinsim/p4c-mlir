#include "gtest/gtest.h"
#include <boost/algorithm/string.hpp>

#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "common.h"
#include "../cfgBuilder.h"


namespace p4mlir::tests {



// This class is used to load CFG either from string or entry BasicBlock
// so we can fuzzy compare two instances of CFGs.
struct TextCFG
{
    struct BB {
        std::vector<std::string> lines;
        int succs = 0;
        bool operator == (const BB& other) const {
            return succs == other.succs && lines == other.lines;
        }
    };
    std::list<BB> basicBlocks;
    TextCFG(const std::string& cfg) { initFromString(cfg); }
    TextCFG(const p4mlir::BasicBlock* entry) : TextCFG(toString(entry)) {}

private:
    void initFromString(const std::string& cfg) {
        std::stringstream ss(cfg);
        std::string line;

        auto tryAdd = [&](std::optional<BB>& x) {
            if (x.has_value()) {
                basicBlocks.push_back(std::move(x.value()));
                x.reset();
            }
        };

        auto parseNumOfSuccessors = [](std::string str) {
            auto pos = std::strlen("successors:");
            int cnt = 0;
            while (pos < str.size()) {
                pos = str.find("bb^", pos);
                if (pos == std::string::npos) {
                    break;
                }
                pos += strlen("bb^");
                ++cnt;
            }
            return cnt;
        };

        std::optional<BB> bb;
        while (std::getline(ss, line)) {
            boost::trim(line);
            if (line.empty()) {
                tryAdd(bb);
                continue;
            }
            if (boost::algorithm::starts_with(line, "bb^")) {
                tryAdd(bb);
                bb.emplace();
                continue;
            }
            BUG_CHECK(bb.has_value(), "Basic block must start with a 'bb^*' label on the first line.");
            if (boost::algorithm::starts_with(line, "successors:")) {
                bb->succs = parseNumOfSuccessors(line);
                continue;
            }
            bb->lines.push_back(line);
        }
        tryAdd(bb);
    }
};

testing::AssertionResult fuzzyEq(TextCFG&& a, TextCFG&& b) {
    auto& ablocks = a.basicBlocks;
    auto& bblocks = b.basicBlocks;
    auto curr = a.basicBlocks.begin();
    while (curr != ablocks.end()) {
        auto next = std::next(curr);
        for (auto it = bblocks.begin(); it != bblocks.end(); ++it) {
            if (*curr == *it) {
                bblocks.erase(it);
                ablocks.erase(curr);
                break;
            }
        }
        curr = next;
    }
    if (ablocks.empty() && bblocks.empty()) {
        return testing::AssertionSuccess();
    }

    auto toString = [](const auto& blocks) {
        std::stringstream ss;
        std::for_each(blocks.begin(), blocks.end(), [&](auto& bb) {
            ss << "bb^_\n";
            for (auto& l : bb.lines) {
                ss << "  " << l << '\n';
            }
            ss << "  successors: " << bb.succs << "\n\n";
        });
        return ss.str();
    };

    return testing::AssertionFailure() << "\n"
                                       << "Not-matched in A:\n"
                                       << toString(ablocks) << "\n"
                                       << "Not-matched in B:\n"
                                       << toString(bblocks);
}

// Checking if two graphs are equal is NP-i problem.
// Let's just check that each node has a lexically equal node
// with the same number of successors.
//
// bb^1
//  f1 = hdr.f2;
//  successors: bb^3 bb^4
//
// equals
//
// bb^2
//  f1 = hdr.f2;
//  successors: bb^5 bb^6
#define CFG_EXPECT_FUZZY_EQ(a, b)                                             \
    do {                                                                      \
        EXPECT_TRUE(fuzzyEq(TextCFG(a), TextCFG(b))); \
    } while (0)

class CFGBuilder : public Test::P4CTest { };


TEST_F(CFGBuilder, Test_multiple_simple_actions) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            hdr.f1 = 3;
            hdr.f2 = hdr.f3;
            hdr.f3 = 5;
            int<16> f4;
            test2.apply();
            return;
        }
        action bar() {
            hdr.f4 = hdr.f1;
            test1.apply();
            hdr.f5 = 3 + hdr.f2 + hdr.f5;
        }
        action baz() {
            checksum.clear();
            checksum.update(hdr.f4);
            hdr.inner.f5 = checksum.get() + hdr.f1;
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)3);
    auto* cfgFoo = getByName(all, "foo");
    auto* cfgBar = getByName(all, "bar");
    auto* cfgBaz = getByName(all, "baz");
    ASSERT_TRUE(cfgFoo);
    ASSERT_TRUE(cfgBar);
    ASSERT_TRUE(cfgBaz);

    CFG_EXPECT_FUZZY_EQ(cfgFoo,
        R"(bb^1
            hdr.f1 = 3;
            hdr.f2 = hdr.f3;
            hdr.f3 = 5;
            int<16> f4;
            test2.apply();
            return;
        )"
    );

    CFG_EXPECT_FUZZY_EQ(cfgBar,
        R"(bb^1
            hdr.f4 = hdr.f1;
            test1.apply();
            hdr.f5 = 3 + hdr.f2 + hdr.f5;
            return;
        )"
    );

    CFG_EXPECT_FUZZY_EQ(cfgBaz,
        R"(bb^1
            checksum.clear();
            checksum.update(hdr.f4);
            hdr.inner.f5 = checksum.get() + hdr.f1;
            return;
        )"
    );
}


TEST_F(CFGBuilder, Test_control_block_with_control_flow) {
    std::string src = P4_SOURCE(R"(
        struct Parsed_packet {}
        struct InControl {}
        struct OutControl {}
        typedef bit<32>  IPv4Address;
        typedef bit<9>  PortId;
        control Pipe<H>(inout H headers,
                in InControl inCtrl,
                out OutControl outCtrl);
        package Pipeline<H>(Pipe<H> p);
        control TopPipe(inout Parsed_packet headers,
                        in InControl inCtrl,
                        out OutControl outCtrl) {
            IPv4Address nextHop;

            action Set_nhop(IPv4Address ipv4_dest, PortId port) {
                nextHop = ipv4_dest;
                headers.ip.ttl = headers.ip.ttl - 1;
                outCtrl.outputPort = port;
            }
            table ipv4_match {
                key = { headers.ip.dstAddr: lpm; }
                actions = { NoAction; Set_nhop; }
                size = 1024;
                default_action = NoAction;
            }
            table check_ttl {
                key = { headers.ip.ttl: exact; }
                actions = { Almost_empty; NoAction; }
                const default_action = NoAction;
            }
            table dmac {
                key = { nextHop: exact; }
                actions = { NoAction; Set_smac; }
                size = 1024;
                default_action = NoAction;
            }
            action Set_smac(IPv4Address smac) {
                if (hdr.inner.port1 == hdr.inner.port2) {
                    if (hdr.inner.port1 == 15) {
                        hdr.inner.f1 = hdr.inner.f1 + 1;
                    }
                    if (hdr.inner.port1 + hdr.f1 == 13) {
                        int<16> f4;
                        hdr.inner.f1 = hdr.f5;
                    } else {
                        hdr.inner.f1 = hdr.f5 + 1;
                    }
                }
                headers.ethernet.srcAddr = smac;
                if (hdr.f1 == 3) {
                    return;
                }
            }
            table smac {
                key = { outCtrl.outputPort: exact; }
                actions = { NoAction; Set_smac; }
                size = 16;
                default_action = NoAction;
            }
            action Empty() {}
            action Almost_empty() { return; }
            apply {
                if (hdr.inner.f3 != 0) {
                    NoAction();
                    return;
                }
                ipv4_match.apply();
                if (outCtrl.outputPort == 2) return;
                check_ttl.apply();
                if (outCtrl.outputPort == 3) {
                    smac.apply();
                    hdr.f1 = hdr.f2 + 3 + hdr.f4;
                    return;
                }
                dmac.apply();
                if (outCtrl.outputPort == hdr.inner.port) return;
                smac.apply();
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)5);
    auto* cfgNhop = getByName(all, "Set_nhop");
    auto cfgSmac = getByName(all, "Set_smac");
    auto* cfgEmpty = getByName(all, "Empty");
    auto* cfgAlmostEmpty = getByName(all, "Almost_empty");
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgNhop);
    ASSERT_TRUE(cfgSmac);
    ASSERT_TRUE(cfgEmpty);
    ASSERT_TRUE(cfgAlmostEmpty);
    ASSERT_TRUE(cfgApply);

    CFG_EXPECT_FUZZY_EQ(cfgNhop,
        R"(bb^1
            nextHop = ipv4_dest;
            headers.ip.ttl = headers.ip.ttl - 1;
            outCtrl.outputPort = port;
            return;
        )"
    );

    CFG_EXPECT_FUZZY_EQ(cfgSmac,
        R"(bb^1
            if (hdr.inner.port1 == hdr.inner.port2)
            successors: bb^2 bb^7

           bb^2
            if (hdr.inner.port1 == 15)
            successors: bb^3 bb^4

           bb^3
            hdr.inner.f1 = hdr.inner.f1 + 1;
            successors: bb^4

           bb^4
            if (hdr.inner.port1 + hdr.f1 == 13)
            successors: bb^5 bb^6

           bb^5
            int<16> f4;
            hdr.inner.f1 = hdr.f5;
            successors: bb^7

           bb^6
            hdr.inner.f1 = hdr.f5 + 1;
            successors: bb^7

           bb^7
            headers.ethernet.srcAddr = smac;
            if (hdr.f1 == 3)
            successors: bb^8 bb^9

           bb^8
            return;

           bb^9
            return;
        )"
    );

    // We insert return statements at the end of actions
    CFG_EXPECT_FUZZY_EQ(cfgEmpty,
        R"(bb^1
            return;
        )"
    );

    CFG_EXPECT_FUZZY_EQ(cfgAlmostEmpty,
        R"(bb^1
            return;
        )"
    );

    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            if (hdr.inner.f3 != 0)
            successors: bb^2 bb^3

           bb^2
            NoAction();
            return;

           bb^3
            ipv4_match.apply();
            if (outCtrl.outputPort == 2)
            successors: bb^4 bb^5

           bb^4
            return;

           bb^5
            check_ttl.apply();
            if (outCtrl.outputPort == 3)
            successors: bb^6 bb^7

           bb^6
            smac.apply();
            hdr.f1 = hdr.f2 + 3 + hdr.f4;
            return;

           bb^7
            dmac.apply();
            if (outCtrl.outputPort == hdr.inner.port)
            successors: bb^8 bb^9

           bb^8
            return;

           bb^9
            smac.apply();
            return;
        )"
    );
}


TEST_F(CFGBuilder, Test_switch_statement) {
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
                switch (test_table.apply().action_run) {
                    foo1 : { hdr.f2 = 3; }
                    foo2 : {
                        bit<16> f4;
                        hdr.f3 = 1;
                        return;
                    }
                    foo3 : { hdr.f4 = hdr.f1; }
                    default: { return; }
                }
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)4);
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgApply);

    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            switch (test_table [foo1][foo2][foo3][default])
            successors: bb^2 bb^3 bb^4 bb^5

          bb^2
            hdr.f2 = 3;
            successors: bb^6

          bb^3
            bit<16> f4;
            hdr.f3 = 1;
            return;

          bb^4
            hdr.f4 = hdr.f1;
            successors: bb^6

          bb^5
            return;

          bb^6
            return;
        )"
    );
}


TEST_F(CFGBuilder, Test_switch_statement_without_default) {
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
                switch (test_table.apply().action_run) {
                    foo1 : { hdr.f2 = 3; }
                    foo2 : { hdr.f3 = 1; return; }
                    foo3 : { hdr.f4 = hdr.f1; }
                }
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)4);
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgApply);

    // Last successor in the list is always the 'none matched'
    // case. Which can be either 'default' or simple fallthrough
    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            switch (test_table [foo1][foo2][foo3])
            successors: bb^2 bb^3 bb^4 bb^5

          bb^2
            hdr.f2 = 3;
            successors: bb^5

          bb^3
            hdr.f3 = 1;
            return;

          bb^4
            hdr.f4 = hdr.f1;
            successors: bb^5

          bb^5
            return;
        )"
    );
}


TEST_F(CFGBuilder, Test_fall_through_switch_statement) {
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
                hdr.f1 = hdr.f2;
                switch (test_table.apply().action_run) {
                    foo1 : { hdr.f2 = 3; }
                    foo2 :
                    foo3 : {
                        hdr.f4 = hdr.f1;
                        if (hdr.f1 == 10) {
                            hdr.f5 = 3;
                        }
                    }
                    default: { hdr.f3 = 9; }
                }
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)4);
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgApply);

    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            hdr.f1 = hdr.f2;
            switch (test_table [foo1][foo2 foo3][default])
            successors: bb^2 bb^3 bb^4

          bb^2
            hdr.f2 = 3;
            successors: bb^6

          bb^3
            hdr.f4 = hdr.f1;
            if (hdr.f1 == 10)
            successors: bb^7 bb^6

          bb^4
            hdr.f3 = 9;
            successors: bb^6

          bb^6
            return;

          bb^7
            hdr.f5 = 3;
            successors: bb^6
        )"
    );
}


TEST_F(CFGBuilder, Test_wierd_fall_through_switch_statement) {
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
                switch (test_table.apply().action_run) {
                    foo1 :
                    foo2 : { hdr.f3 = 1; }
                    foo3 :
                    default :
                }
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)4);
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgApply);

    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            switch (test_table [foo1 foo2][foo3 default])
            successors: bb^2 bb^3

          bb^2
            hdr.f3 = 1;
            successors: bb^3

          bb^3
            return;
        )"
    );
}


TEST_F(CFGBuilder, Test_wierd_fall_through_switch_statement_without_default) {
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
                switch (test_table.apply().action_run) {
                    foo1 :
                    foo2 : { hdr.f3 = 1; }
                    foo3 :
                }
            }
        }
        Pipeline(TopPipe()) main;
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), (std::size_t)4);
    auto* cfgApply = getByName(all, "TopPipe");
    ASSERT_TRUE(cfgApply);

    CFG_EXPECT_FUZZY_EQ(cfgApply,
        R"(bb^1
            switch (test_table [foo1 foo2][foo3])
            successors: bb^2 bb^3

          bb^2
            hdr.f3 = 1;
            successors: bb^3

          bb^3
            return;
        )"
    );
}


// TEST_F(CFGBuilder, Test_switch_statement_with_empty_cases) {
//     std::string src = P4_SOURCE(R"(
//         struct Parsed_packet {}
//         control Pipe<H>(inout H headers);
//         package Pipeline<H>(Pipe<H> p);
//         control TopPipe(inout Parsed_packet hdr) {
//             action foo1() {}
//             action foo2() {}
//             action foo3() {}
//             action foo4() {}
//             action foo5() {}
//             table test_table {
//                 key = { hdr.f1: exact; }
//                 actions = { foo1; foo2; foo3; foo4; foo5; }
//                 size = 1024;
//             }
//             apply {
//                 switch (test_table.apply().action_run) {
//                     foo1 : { hdr.f2 = 3; }
//                     foo2 : {}
//                     foo3 : { hdr.f4 = hdr.f1; }
//                     foo4 :
//                     foo5 : {}
//                     default: {}
//                 }
//             }
//         }
//         Pipeline(TopPipe()) main;
//     )");
//     auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
//     ASSERT_TRUE(pgm && ::errorCount() == 0);

//     auto b = new p4mlir::CFGBuilder;
//     pgm->apply(*b);
//     auto all = b->getCFG();

//     ASSERT_EQ(all.size(), (std::size_t)4);
//     auto* cfgApply = getByName(all, "TopPipe");
//     ASSERT_TRUE(cfgApply);

//     CFG_EXPECT_FUZZY_EQ(cfgApply,
//         R"(bb^1
//             switch (test_table [foo1][foo2][foo3][foo4 foo5][default])
//             successors: bb^2 bb^3 bb^4 bb^5 bb^6 // What to do here?

//           bb^2
//             hdr.f2 = 3;
//             successors: bb^7

//           bb^3
//             hdr.f4 = hdr.f1;
//             successors: bb^7

//           bb^4
//             return;
//         )"
//     );
// }


} // namespace p4mlir::tests