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
        transition select (arg2) {
                   arg1: parse_ipv4;
                   7: reject;
                   _: accept;
        }
    }

    state with_default {
        transition select (arg2) {
                   5: parse_ipv4;
                   7: reject;
                   default: parse_ipv4;
        }
    }

    state parse_ipv4 {
        transition select(b.length()) {
                    0: start;
                    1: drop;
                    2: foo1;
                    3: foo2;
                    5: foo3;
        }
    }

    state foo1 {
        bool flag = false;
        transition select(flag) {
                    false: accept;
                    default: parse_ipv4;
                    true : parse_ipv4;
        }
    }

    state foo2 {}
    state foo3 {}

    state drop {
        transition reject;
    }
}

// TODO: If no label matches, the execution triggers a runtime error with the standard error code error.NoMatch.

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
// CHECK-NEXT:       %1 = p4.load(%arg2) : !p4.ref<si32> -> si32
// CHECK-NEXT:       %2 = p4.tuple(%1) : (si32) -> tuple<si32>
// CHECK-NEXT:       p4.select_transition %2 : tuple<si32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             p4.init si32 (%arg1) : (si32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@parse_ipv4(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 7 : si32
// CHECK-NEXT:             p4.init si32 (%3) : (si32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.parser_reject
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @with_default(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.load(%arg2) : !p4.ref<si32> -> si32
// CHECK-NEXT:       %2 = p4.tuple(%1) : (si32) -> tuple<si32>
// CHECK-NEXT:       p4.select_transition %2 : tuple<si32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 5 : si32
// CHECK-NEXT:             p4.init si32 (%3) : (si32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@parse_ipv4(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 7 : si32
// CHECK-NEXT:             p4.init si32 (%3) : (si32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.parser_reject
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.transition @TopParser::@parse_ipv4(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @parse_ipv4(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.call_method %arg0 @packet_in::@length_0() : (!p4.ref<!p4.extern_class<"packet_in">>) -> ui32
// CHECK-NEXT:       %2 = p4.tuple(%1) : (ui32) -> tuple<ui32>
// CHECK-NEXT:       p4.select_transition %2 : tuple<ui32> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 0 : ui32
// CHECK-NEXT:             p4.init ui32 (%3) : (ui32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@start(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 1 : ui32
// CHECK-NEXT:             p4.init ui32 (%3) : (ui32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@drop(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 2 : ui32
// CHECK-NEXT:             p4.init ui32 (%3) : (ui32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo1(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 3 : ui32
// CHECK-NEXT:             p4.init ui32 (%3) : (ui32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo2(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %3 = p4.constant 5 : ui32
// CHECK-NEXT:             p4.init ui32 (%3) : (ui32)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@foo3(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.parser_reject
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo1(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       %1 = p4.constant false
// CHECK-NEXT:       %2 = p4.copy(%1) : i1 -> i1
// CHECK-NEXT:       %3 = p4.tuple(%2) : (i1) -> tuple<i1>
// CHECK-NEXT:       p4.select_transition %3 : tuple<i1> {
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %4 = p4.constant false
// CHECK-NEXT:             p4.init i1 (%4) : (i1)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.parser_accept
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_default_case {
// CHECK-NEXT:           p4.transition @TopParser::@parse_ipv4(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:         p4.select_transition_case {
// CHECK-NEXT:           p4.select_transition_keys {
// CHECK-NEXT:             %4 = p4.constant true
// CHECK-NEXT:             p4.init i1 (%4) : (i1)
// CHECK-NEXT:           }
// CHECK-NEXT:           p4.transition @TopParser::@parse_ipv4(%arg3, %arg4) : (!p4.ref<ui32>, !p4.ref<i1>)
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo2(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       p4.parser_reject
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @foo3(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       p4.parser_reject
// CHECK-NEXT:     }
// CHECK-NEXT:     p4.state @drop(%arg3: !p4.ref<ui32>, %arg4: !p4.ref<i1>) {
// CHECK-NEXT:       %0 = p4.self : !p4.ref<!p4.parser<"TopParser">>
// CHECK-NEXT:       p4.parser_reject
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

