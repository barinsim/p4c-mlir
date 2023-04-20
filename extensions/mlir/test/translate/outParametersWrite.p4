// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<10> f1;
}

action foo(inout int<10> arg1, out int<10> arg2, inout MyHeader hdr_arg, int<10> arg3) {
    int<10> x1 = arg1;
    arg1 = x1 + 1;
    arg2 = arg1 + 1;
    arg2 = arg3;
    arg1 = arg3;
    MyHeader hdr_local;
    hdr_arg = hdr_local;
    hdr_local = hdr_arg;
}

// CHECK: p4.action @foo(%arg0: !p4.ref<si10>, %arg1: !p4.ref<si10>, %arg2: !p4.ref<!p4.header<"MyHeader">>, %arg3: si10) {
    // CHECK-NEXT: %0 = p4.load(%arg0) : !p4.ref<si10> -> si10
    // CHECK-NEXT: %1 = p4.copy(%0) : si10 -> si10
    // CHECK-NEXT: %2 = p4.constant 1 : si10
    // CHECK-NEXT: %3 = p4.add(%1, %2) : (si10, si10) -> si10
    // CHECK-NEXT: p4.store(%arg0, %3) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %4 = p4.load(%arg0) : !p4.ref<si10> -> si10
    // CHECK-NEXT: %5 = p4.constant 1 : si10
    // CHECK-NEXT: %6 = p4.add(%4, %5) : (si10, si10) -> si10
    // CHECK-NEXT: p4.store(%arg1, %6) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: p4.store(%arg1, %arg3) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: p4.store(%arg0, %arg3) : (!p4.ref<si10>, si10) -> ()
    // CHECK-NEXT: %7 = p4.uninitialized : !p4.header<"MyHeader">
    // CHECK-NEXT: %8 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: p4.store(%8, %7) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: %9 = p4.load(%8) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
    // CHECK-NEXT: p4.store(%arg2, %9) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: %10 = p4.load(%arg2) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
    // CHECK-NEXT: p4.store(%8, %10) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }

