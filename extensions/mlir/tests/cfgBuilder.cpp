#include "gtest/gtest.h"
#include <boost/algorithm/string.hpp>

#include "test/gtest/helpers.h"
#include "frontends/common/parseInput.h"
#include "../cfgBuilder.h"


namespace p4mlir::tests {



// This class is used to load CFG either from string or entry BasicBlock
// so we can fuzzy compare two instances of CFGs.
struct CFGTestIntermediary
{
    struct BB {
        std::vector<std::string> lines;
        int succs = 0;
        bool operator == (const BB& other) const {
            return succs == other.succs && lines == other.lines;
        }
    };
    std::list<BB> basicBlocks;
    CFGTestIntermediary(const std::string& cfg) { initFromString(cfg); }
    CFGTestIntermediary(const p4mlir::BasicBlock* entry) : CFGTestIntermediary(toString(entry)) {}

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

testing::AssertionResult fuzzyEq(CFGTestIntermediary&& a, CFGTestIntermediary&& b) {
    auto& ablocks = a.basicBlocks;
    auto& bblocks = b.basicBlocks;
    while (!ablocks.empty()) {
        auto& curr = ablocks.front();
        bool success = false;
        for (auto it = bblocks.begin(); it != bblocks.end(); ++it) {
            if (curr == *it) {
                success = true;
                bblocks.erase(it);
                break;
            }
        }
        if (!success) {
            break;
        }
        ablocks.pop_front();
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
                                       << toString(ablocks) << "\n\n"
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
#define CFG_EXPECT_FUZZY_EQ(a, b)                                               \
    do {                                                                        \
        EXPECT_TRUE(fuzzyEq(CFGTestIntermediary(a), CFGTestIntermediary(b)));  \
    } while (0)



class CFGBuilder : public Test::P4CTest { };


TEST_F(CFGBuilder, Test_CFG_of_simple_action) {
    std::string src = P4_SOURCE(R"(
        action foo() {
            hdr.f1 = 3;
            hdr.f2 = hdr.f3;
            hdr.f3 = 5;
        }
    )");
    auto* pgm = P4::parseP4String(src, CompilerOptions::FrontendVersion::P4_16);
    ASSERT_TRUE(pgm && ::errorCount() == 0);

    auto b = new p4mlir::CFGBuilder;
    pgm->apply(*b);
    auto all = b->getCFG();

    ASSERT_EQ(all.size(), 1);
    auto& [decl, cfg] = *all.begin();
    EXPECT_EQ(decl->getName(), "foo");

    CFG_EXPECT_FUZZY_EQ(cfg,
        R"(bb^1
            hdr.f1 = 3;
            hdr.f2 = hdr.f3;
            hdr.f3 = 5;
        )"
    );
}


} // namespace p4mlir::tests