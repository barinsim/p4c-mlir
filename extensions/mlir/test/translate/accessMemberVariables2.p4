// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    bit<16> f2;
    bit<10> f3;
}

action bar(out int<16> arg1, inout bit<16> arg2, in bit<16> arg3) {
    return;
}

action foo(inout MyHeader arg1) {
    int<16> x1 = 2;
    MyHeader x2;
    MyHeader x3;
    arg1.f1 = x1;
    arg1.f2 = 3;
    x1 = arg1.f1 + 2;
    x3.f1 = arg1.f1;
    x3.f1 = x2.f1;
    x3.f2 = 2;
    x3.f3 = 3;
    bar(arg1.f1, x3.f2, x2.f2);
}

// CHECK: p4.action @foo(%arg0: !p4.ref<!p4.header<"MyHeader">>) {
    // CHECK-NEXT: %0 = p4.constant 2 : si16
    // CHECK-NEXT: %1 = p4.cast(%0) : si16 -> si16
    // CHECK-NEXT: %2 = p4.copy(%1) : si16 -> si16
    // CHECK-NEXT: %3 = p4.uninitialized : !p4.header<"MyHeader">
    // CHECK-NEXT: %4 = p4.uninitialized : !p4.header<"MyHeader">
    // CHECK-NEXT: %5 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: p4.store(%5, %4) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: %6 = p4.get_member_ref(%arg0) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: p4.store(%6, %2) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: %7 = p4.get_member_ref(%arg0) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
    // CHECK-NEXT: %8 = p4.constant 3 : ui16
    // CHECK-NEXT: %9 = p4.cast(%8) : ui16 -> ui16
    // CHECK-NEXT: p4.store(%7, %9) : (!p4.ref<ui16>, ui16) -> ()
    // CHECK-NEXT: %10 = p4.get_member_ref(%arg0) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %11 = p4.load(%10) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %12 = p4.constant 2 : si16
    // CHECK-NEXT: %13 = p4.add(%11, %12) : (si16, si16) -> si16
    // CHECK-NEXT: %14 = p4.copy(%13) : si16 -> si16
    // CHECK-NEXT: %15 = p4.get_member_ref(%5) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %16 = p4.get_member_ref(%arg0) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %17 = p4.load(%16) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%15, %17) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: %18 = p4.get_member_ref(%5) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %19 = p4.get_member(%3) "f1" : !p4.header<"MyHeader"> -> si16
    // CHECK-NEXT: p4.store(%18, %19) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: %20 = p4.get_member_ref(%5) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
    // CHECK-NEXT: %21 = p4.constant 2 : ui16
    // CHECK-NEXT: %22 = p4.cast(%21) : ui16 -> ui16
    // CHECK-NEXT: p4.store(%20, %22) : (!p4.ref<ui16>, ui16) -> ()
    // CHECK-NEXT: %23 = p4.get_member_ref(%5) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
    // CHECK-NEXT: %24 = p4.constant 3 : ui10
    // CHECK-NEXT: %25 = p4.cast(%24) : ui10 -> ui10
    // CHECK-NEXT: p4.store(%23, %25) : (!p4.ref<ui10>, ui10) -> ()
    // CHECK-NEXT: %26 = p4.get_member_ref(%arg0) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %27 = p4.alloc : !p4.ref<si16>
    // CHECK-NEXT: %28 = p4.get_member_ref(%5) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
    // CHECK-NEXT: %29 = p4.alloc : !p4.ref<ui16>
    // CHECK-NEXT: %30 = p4.load(%28) : !p4.ref<ui16> -> ui16
    // CHECK-NEXT: p4.store(%29, %30) : (!p4.ref<ui16>, ui16) -> ()
    // CHECK-NEXT: %31 = p4.get_member(%3) "f2" : !p4.header<"MyHeader"> -> ui16
    // CHECK-NEXT: p4.call @bar(%27, %29, %31) : (!p4.ref<si16>, !p4.ref<ui16>, ui16) -> ()
    // CHECK-NEXT: %32 = p4.load(%27) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%26, %32) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: %33 = p4.load(%29) : !p4.ref<ui16> -> ui16
    // CHECK-NEXT: p4.store(%28, %33) : (!p4.ref<ui16>, ui16) -> ()
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }


