// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<10> f1;
}

action bar(inout int<10> arg1, out int<10> arg2, out MyHeader arg3, int<10> arg4) {
    return;
}

action foo(inout int<10> arg1, out MyHeader arg2, out int<10> arg3, in int<10> arg4) {
    int<10> x1 = 2;
    bar(arg1, arg1, arg2, arg1);
    bar(arg3, arg3, arg2, arg4);
}

// CHECK: p4.action @bar(%arg0: !p4.ref<si10>, %arg1: !p4.ref<si10>, %arg2: !p4.ref<!p4.header<"MyHeader">>, %arg3: si10) {
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK: p4.action @foo(%arg0: !p4.ref<si10>, %arg1: !p4.ref<!p4.header<"MyHeader">>, %arg2: !p4.ref<si10>, %arg3: si10) {
    // CHECK-NEXT: %0 = p4.constant 2 : si10
    // CHECK-NEXT: %1 = p4.cast(%0) : si10 -> si10
    // CHECK-NEXT: %2 = p4.copy(%1) : si10 -> si10
    // CHECK-NEXT: %3 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: %4 = p4.load(%arg0) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%3, %4) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %5 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: %6 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: %7 = p4.load(%arg0) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.call @bar(%3, %5, %6, %7) : (!p4.ref<si10>, !p4.ref<si10>, !p4.ref<!p4.header<"MyHeader">>, si10) -> ()
    // CHECK-NEXT: %8 = p4.load(%3) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%arg0, %8) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %9 = p4.load(%5) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%arg0, %9) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %10 = p4.load(%6) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
    // CHECK-NEXT: p4.store(%arg1, %10) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: %11 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: %12 = p4.load(%arg2) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%11, %12) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %13 = p4.alloc : !p4.ref<si10>
    // CHECK-NEXT: %14 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: p4.call @bar(%11, %13, %14, %arg3) : (!p4.ref<si10>, !p4.ref<si10>, !p4.ref<!p4.header<"MyHeader">>, si10) -> ()
    // CHECK-NEXT: %15 = p4.load(%11) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%arg2, %15) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %16 = p4.load(%13) : !p4.ref<si10> -> si10
    // CHECK-NEXT: p4.store(%arg2, %16) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %17 = p4.load(%14) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
    // CHECK-NEXT: p4.store(%arg1, %17) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }
