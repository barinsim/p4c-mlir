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
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %3 = p4.get_member_ref(%2) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%3, %0) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %4 = p4.get_member_ref(%2) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%4, %1) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %5 = p4.get_member_ref(%2) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %6 = p4.constant true
// CHECK-NEXT: p4.store(%5, %6) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %7 = p4.load(%2) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %8 = p4.copy(%7) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %9 = p4.constant 1 : si16
// CHECK-NEXT: %10 = p4.cast(%9) : si16 -> si16
// CHECK-NEXT: %11 = p4.copy(%10) : si16 -> si16
// CHECK-NEXT: %12 = p4.constant 2 : ui16
// CHECK-NEXT: %13 = p4.cast(%12) : ui16 -> ui16
// CHECK-NEXT: %14 = p4.copy(%13) : ui16 -> ui16
// CHECK-NEXT: %15 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %16 = p4.get_member_ref(%15) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%16, %11) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %17 = p4.get_member_ref(%15) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%17, %14) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %18 = p4.get_member_ref(%15) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %19 = p4.constant true
// CHECK-NEXT: p4.store(%18, %19) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %20 = p4.load(%15) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %21 = p4.copy(%20) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %22 = p4.get_member(%21) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: %23 = p4.get_member(%21) "f2" : !p4.header<"MyHeader"> -> ui16
// CHECK-NEXT: %24 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %25 = p4.get_member_ref(%24) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%25, %22) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %26 = p4.get_member_ref(%24) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%26, %23) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %27 = p4.get_member_ref(%24) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %28 = p4.constant true
// CHECK-NEXT: p4.store(%27, %28) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %29 = p4.load(%24) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %30 = p4.copy(%29) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %31 = p4.copy(%30) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.return
// CHECK-NEXT: }


