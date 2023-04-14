// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

action foo(in int<16> arg1, inout bit<10> arg2, out bit<10> arg3, inout MyHeader hdr, bool arg4) {
    return;
}

// CHECK: p4.action @foo(%arg0: si16, %arg1: !p4.ref<ui10>, %arg2: !p4.ref<ui10>, %arg3: !p4.ref<!p4.header<"MyHeader">>, %arg4: i1) {
    // CHECK-NEXT: p4.return
// CHECK-NEXT: }
