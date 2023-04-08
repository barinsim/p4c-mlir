// RUN: p4c-mlir-translate %s | FileCheck %s

// CHECK: p4.header @MyHeader {
header MyHeader {
	// CHECK-NEXT: p4.field_decl @f1 : si16
	int<16> f1;
	// CHECK-NEXT: p4.field_decl @f2 : ui16
    bit<16> f2;
// CHECK-NEXT: }
}

// CHECK-NEXT:  p4.action @foo() {
action foo() {
    // CHECK-NEXT: %0 = p4.uninitialized : !p4.header<"MyHeader">
    MyHeader hdr;
    // CHECK-NEXT: p4.return
    return;
// CHECK-NEXT: }
}
