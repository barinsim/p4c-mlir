// RUN: p4c-mlir-translate %s | FileCheck %s

parser TopParser(in bit<32> arg1, inout bit<32> arg2) {
    bit<32> val1 = 2;

    state start {
        transition select (arg1) {
                   arg1 &&& 3: foo;
                   (arg1 &&& 3): foo;
                   2 &&& arg1: foo;
                   2 &&& 3: foo;
                   arg1 &&& arg2: accept;
        }
    }

    state foo {
        bit<32> local_val = 2;
        if (local_val == 1) {
            local_val = 3;
        } else {
            local_val = 4;
        }
        transition select (arg1, arg2) {
                   (2..3, arg1 &&& 4): start;
                   (val1 &&& local_val, arg1 &&& 4): start;
        }
    }
}

// CHECK: module {
// CHECK-NEXT:   p4.parser @TopParser(%arg0: ui32, %arg1: !p4.ref<ui32>) {
// CHECK-NEXT:     p4.state @start(%arg2: !p4.ref<ui32>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.tuple(%arg0) : (ui32) -> tuple<ui32>
// CHECK-NEXT:       p4.select_transition %1 : tuple<ui32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 3 : si64
// CHECK-NEXT:             %3 = p4.cast(%2) : si64 -> ui32
// CHECK-NEXT:             %4 = p4.mask(%arg0, %3) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             p4.init !p4.set<ui32> (%4) : (!p4.set<ui32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 3 : si64
// CHECK-NEXT:             %3 = p4.cast(%2) : si64 -> ui32
// CHECK-NEXT:             %4 = p4.mask(%arg0, %3) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %5 = p4.set_product(%4) : (!p4.set<ui32>) -> !p4.set<tuple<ui32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<ui32>> (%5) : (!p4.set<tuple<ui32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 2 : si64
// CHECK-NEXT:             %3 = p4.cast(%2) : si64 -> ui32
// CHECK-NEXT:             %4 = p4.mask(%3, %arg0) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             p4.init !p4.set<ui32> (%4) : (!p4.set<ui32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 2 : ui32
// CHECK-NEXT:             %3 = p4.constant 3 : ui32
// CHECK-NEXT:             %4 = p4.mask(%2, %3) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             p4.init !p4.set<ui32> (%4) : (!p4.set<ui32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.load(%arg1) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:             %3 = p4.mask(%arg0, %2) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             p4.init !p4.set<ui32> (%3) : (!p4.set<ui32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_reject with error @NoMatch
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo(%arg2: !p4.ref<ui32>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.constant 2 : ui32
// CHECK-NEXT:       %2 = p4.cast(%1) : ui32 -> ui32
// CHECK-NEXT:       %3 = p4.copy(%2) : ui32 -> ui32
// CHECK-NEXT:       %4 = p4.constant 1 : si64
// CHECK-NEXT:       %5 = p4.cast(%4) : si64 -> ui32
// CHECK-NEXT:       %6 = p4.cmp(%3, %5) eq : (ui32, ui32) -> i1
// CHECK-NEXT:       cf.cond_br %6, ^bb1, ^bb2
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %7 = p4.constant 3 : ui32
// CHECK-NEXT:       %8 = p4.cast(%7) : ui32 -> ui32
// CHECK-NEXT:       %9 = p4.copy(%8) : ui32 -> ui32
// CHECK-NEXT:       cf.br ^bb3(%9 : ui32)
// CHECK-NEXT:     ^bb2:  // pred: ^bb0
// CHECK-NEXT:       %10 = p4.constant 4 : ui32
// CHECK-NEXT:       %11 = p4.cast(%10) : ui32 -> ui32
// CHECK-NEXT:       %12 = p4.copy(%11) : ui32 -> ui32
// CHECK-NEXT:       cf.br ^bb3(%12 : ui32)
// CHECK-NEXT:     ^bb3(%13: ui32):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:       %14 = p4.load(%arg1) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:       %15 = p4.tuple(%arg0, %14) : (ui32, ui32) -> tuple<ui32, ui32>
// CHECK-NEXT:       p4.select_transition %15 : tuple<ui32, ui32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %16 = p4.constant 2 : ui32
// CHECK-NEXT:             %17 = p4.constant 3 : ui32
// CHECK-NEXT:             %18 = p4.range(%16, %17) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %19 = p4.constant 4 : si64
// CHECK-NEXT:             %20 = p4.cast(%19) : si64 -> ui32
// CHECK-NEXT:             %21 = p4.mask(%arg0, %20) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %22 = p4.set_product(%18, %21) : (!p4.set<ui32>, !p4.set<ui32>) -> !p4.set<tuple<ui32, ui32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<ui32, ui32>> (%22) : (!p4.set<tuple<ui32, ui32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@start(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %16 = p4.load(%arg2) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:             %17 = p4.mask(%16, %13) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %18 = p4.constant 4 : si64
// CHECK-NEXT:             %19 = p4.cast(%18) : si64 -> ui32
// CHECK-NEXT:             %20 = p4.mask(%arg0, %19) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %21 = p4.set_product(%17, %20) : (!p4.set<ui32>, !p4.set<ui32>) -> !p4.set<tuple<ui32, ui32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<ui32, ui32>> (%21) : (!p4.set<tuple<ui32, ui32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@start(%arg2) : (!p4.ref<ui32>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_reject with error @NoMatch
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.apply {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.constant 2 : ui32
// CHECK-NEXT:       %2 = p4.cast(%1) : ui32 -> ui32
// CHECK-NEXT:       %3 = p4.alloc : !p4.ref<ui32>
// CHECK-NEXT:       p4.store(%3, %2) : (!p4.ref<ui32>, ui32) -> ()
// CHECK-NEXT:       p4.transition @TopParser::@start(%3) : (!p4.ref<ui32>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
