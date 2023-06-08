// RUN: p4c-mlir-translate %s | FileCheck %s

extern packet_in {
    void extract<T>(out T hdr);
    void extract<T>(out T variableSizeHeader,
                    in bit<32> variableFieldSizeInBits);
    T lookahead<T>();
    void advance(in bit<32> sizeInBits);
    bit<32> length();
}

parser TopParser(packet_in b, in int<32> arg1, inout int<32> arg2) {
    bit<32> val1 = 2;
    bool flag;

    state start {
        transition select (arg1, {val1, val1, val1}) {
                   (arg1, {1, 2, 3}): foo1;
                   (3..7, _): foo2;
                   (_ , _): accept;
        }
    }

    state foo1 {
        transition select (arg1) {
                   4..10: foo2;
                   (4..10): foo2;
                   2..arg2: foo3;
                   (2..arg2): foo3;
                   1: reject;
                   _: accept;
        }
    }

    state foo2 {
        transition select (arg1, val1) {
                   (1, 3..4): foo2;
                   _: accept;
        }
    }

    state foo3 {
        bool local_flag = flag;
        if (flag == false) {
            local_flag = false;
        } else {
            local_flag = true;
        }
        transition select (local_flag) {
                   false: foo2;
                   true: foo1;
        }
    }
}

// CHECK: module {
// CHECK-NEXT:   p4.extern_class @packet_in {
// CHECK-NEXT:     p4.extern @extract_1<@T>(!p4.ref<!p4.type_var<@T>>)
// CHECK-NEXT:     p4.extern @extract_2<@T>(!p4.ref<!p4.type_var<@T>>, ui32)
// CHECK-NEXT:     p4.extern @lookahead_0<@T>() -> !p4.type_var<@T>
// CHECK-NEXT:     p4.extern @advance_1(ui32)
// CHECK-NEXT:     p4.extern @length_0() -> ui32
// CHECK-NEXT:   }
// CHECK-NEXT:   p4.parser @TopParser(%arg0: !p4.ref<!p4.extern_class<"packet_in">>, %arg1: si32, %arg2: !p4.ref<si32>) {
// CHECK-NEXT:     p4.state @start(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.load(%arg3) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:       %2 = p4.load(%arg3) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:       %3 = p4.load(%arg3) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:       %4 = p4.tuple(%1, %2, %3) : (ui32, ui32, ui32) -> tuple<ui32, ui32, ui32>
// CHECK-NEXT:       %5 = p4.tuple(%arg1, %4) : (si32, tuple<ui32, ui32, ui32>) -> tuple<si32, tuple<ui32, ui32, ui32>>
// CHECK-NEXT:       p4.select_transition %5 : tuple<si32, tuple<ui32, ui32, ui32>> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %6 = p4.constant 1 : ui32
// CHECK-NEXT:             %7 = p4.constant 2 : ui32
// CHECK-NEXT:             %8 = p4.constant 3 : ui32
// CHECK-NEXT:             %9 = p4.set_product(%6, %7, %8) : (ui32, ui32, ui32) -> !p4.set<tuple<ui32, ui32, ui32>>
// CHECK-NEXT:             %10 = p4.set_product(%arg1, %9) : (si32, !p4.set<tuple<ui32, ui32, ui32>>) -> !p4.set<tuple<si32, tuple<ui32, ui32, ui32>>>
// CHECK-NEXT:             p4.init tuple<si32, tuple<ui32, ui32, ui32>> (%10) : (!p4.set<tuple<si32, tuple<ui32, ui32, ui32>>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo1(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %6 = p4.constant 3 : si32
// CHECK-NEXT:             %7 = p4.constant 7 : si32
// CHECK-NEXT:             %8 = p4.range(%6, %7) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             %9 = p4.dontcare : !p4.dontcare
// CHECK-NEXT:             %10 = p4.set_product(%8, %9) : (!p4.set<si32>, !p4.dontcare) -> !p4.set<tuple<si32, !p4.dontcare>>
// CHECK-NEXT:             p4.init !p4.set<tuple<si32, !p4.dontcare>> (%10) : (!p4.set<tuple<si32, !p4.dontcare>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo1(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.tuple(%arg1) : (si32) -> tuple<si32>
// CHECK-NEXT:       p4.select_transition %1 : tuple<si32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 4 : si32
// CHECK-NEXT:             %3 = p4.constant 10 : si32
// CHECK-NEXT:             %4 = p4.range(%2, %3) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             p4.init !p4.set<si32> (%4) : (!p4.set<si32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 4 : si32
// CHECK-NEXT:             %3 = p4.constant 10 : si32
// CHECK-NEXT:             %4 = p4.range(%2, %3) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             %5 = p4.set_product(%4) : (!p4.set<si32>) -> !p4.set<tuple<si32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<si32>> (%5) : (!p4.set<tuple<si32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 2 : si64
// CHECK-NEXT:             %3 = p4.cast(%2) : si64 -> si32
// CHECK-NEXT:             %4 = p4.load(%arg2) : !p4.ref<si32> -> si32
// CHECK-NEXT:             %5 = p4.range(%3, %4) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             p4.init !p4.set<si32> (%5) : (!p4.set<si32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo3(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 2 : si64
// CHECK-NEXT:             %3 = p4.cast(%2) : si64 -> si32
// CHECK-NEXT:             %4 = p4.load(%arg2) : !p4.ref<si32> -> si32
// CHECK-NEXT:             %5 = p4.range(%3, %4) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             %6 = p4.set_product(%5) : (!p4.set<si32>) -> !p4.set<tuple<si32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<si32>> (%6) : (!p4.set<tuple<si32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo3(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %2 = p4.constant 1 : si32
// CHECK-NEXT:             %3 = p4.range(%2, %2) : (si32, si32) -> !p4.set<si32>
// CHECK-NEXT:             p4.init si32 (%3) : (!p4.set<si32>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.parser_reject
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo2(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.load(%arg3) : !p4.ref<ui32> -> ui32
// CHECK-NEXT:       %2 = p4.tuple(%arg1, %1) : (si32, ui32) -> tuple<si32, ui32>
// CHECK-NEXT:       p4.select_transition %2 : tuple<si32, ui32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 1 : si32
// CHECK-NEXT:             %4 = p4.constant 3 : ui32
// CHECK-NEXT:             %5 = p4.constant 4 : ui32
// CHECK-NEXT:             %6 = p4.range(%4, %5) : (ui32, ui32) -> !p4.set<ui32>
// CHECK-NEXT:             %7 = p4.set_product(%3, %6) : (si32, !p4.set<ui32>) -> !p4.set<tuple<si32, ui32>>
// CHECK-NEXT:             p4.init !p4.set<tuple<si32, ui32>> (%7) : (!p4.set<tuple<si32, ui32>>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo3(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.load(%arg4) : !p4.ref<i1> -> i1
// CHECK-NEXT:       %2 = p4.copy(%1) : i1 -> i1
// CHECK-NEXT:       %3 = p4.load(%arg4) : !p4.ref<i1> -> i1
// CHECK-NEXT:       %4 = p4.constant false
// CHECK-NEXT:       %5 = p4.cmp(%3, %4) eq : (i1, i1) -> i1
// CHECK-NEXT:       cf.cond_br %5, ^bb1, ^bb2
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %6 = p4.constant false
// CHECK-NEXT:       %7 = p4.copy(%6) : i1 -> i1
// CHECK-NEXT:       cf.br ^bb3(%7 : i1)
// CHECK-NEXT:     ^bb2:  // pred: ^bb0
// CHECK-NEXT:       %8 = p4.constant true
// CHECK-NEXT:       %9 = p4.copy(%8) : i1 -> i1
// CHECK-NEXT:       cf.br ^bb3(%9 : i1)
// CHECK-NEXT:     ^bb3(%10: i1):  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT:       %11 = p4.tuple(%10) : (i1) -> tuple<i1>
// CHECK-NEXT:       p4.select_transition %11 : tuple<i1> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %12 = p4.constant false
// CHECK-NEXT:             %13 = p4.range(%12, %12) : (i1, i1) -> !p4.set<i1>
// CHECK-NEXT:             p4.init i1 (%13) : (!p4.set<i1>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %12 = p4.constant true
// CHECK-NEXT:             %13 = p4.range(%12, %12) : (i1, i1) -> !p4.set<i1>
// CHECK-NEXT:             p4.init i1 (%13) : (!p4.set<i1>)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo1(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
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
// CHECK-NEXT:       %4 = p4.uninitialized : i1
// CHECK-NEXT:       %5 = p4.alloc : !p4.ref<i1>
// CHECK-NEXT:       p4.store(%5, %4) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT:       p4.transition @TopParser::@start(%3, %5) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

