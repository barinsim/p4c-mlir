// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    bit<16> f2;
}

action foo() {
    MyHeader hdr1 = {1, 2};

    int<16> x1 = 1;
    bit<16> x2 = 2;
    MyHeader hdr2 = {x1, x2};

    MyHeader hdr3 = {hdr2.f1, hdr2.f2};

    MyHeader hdr4 = hdr3;
}

// CHECK: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.member_decl @f2 : ui16
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.constant 1 : si16
// CHECK-NEXT: %1 = p4.constant 2 : ui16
// CHECK-NEXT: %2 = p4.constant true
// CHECK-NEXT: %3 = p4.tuple(%0, %1, %2) : (si16, ui16, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %4 = p4.copy(%3) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %5 = p4.constant 1 : si16
// CHECK-NEXT: %6 = p4.cast(%5) : si16 -> si16
// CHECK-NEXT: %7 = p4.copy(%6) : si16 -> si16
// CHECK-NEXT: %8 = p4.constant 2 : ui16
// CHECK-NEXT: %9 = p4.cast(%8) : ui16 -> ui16
// CHECK-NEXT: %10 = p4.copy(%9) : ui16 -> ui16
// CHECK-NEXT: %11 = p4.constant true
// CHECK-NEXT: %12 = p4.tuple(%7, %10, %11) : (si16, ui16, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %13 = p4.copy(%12) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %14 = p4.get_member(%13) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: %15 = p4.get_member(%13) "f2" : !p4.header<"MyHeader"> -> ui16
// CHECK-NEXT: %16 = p4.constant true
// CHECK-NEXT: %17 = p4.tuple(%14, %15, %16) : (si16, ui16, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %18 = p4.copy(%17) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %19 = p4.copy(%18) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.return
// CHECK-NEXT: }


