// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
	int<16> f1;
	int<16> f2;
}

action foo() {
    MyHeader hdr;
    hdr.setValid();
    hdr.setInvalid();
    if (hdr.isValid()) {
        hdr.f1 = hdr.f2;
    }
    bool val = hdr.isValid();
    if (val) {
        val = false;
    }
    return;
}

action bar() {
    MyHeader hdr;
    hdr.isValid();
}

// CHECK: p4.header @MyHeader {
    // CHECK-NEXT: p4.member_decl @f1 : si16
    // CHECK-NEXT: p4.member_decl @f2 : si16
    // CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %1 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%1, %0) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %2 = p4.get_member_ref(%1) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %3 = p4.constant true
// CHECK-NEXT: p4.store(%2, %3) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %4 = p4.get_member_ref(%1) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %5 = p4.constant false
// CHECK-NEXT: p4.store(%4, %5) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %6 = p4.get_member_ref(%1) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %7 = p4.load(%6) : !p4.ref<i1> -> i1
// CHECK-NEXT: cf.cond_br %7, ^bb1, ^bb2

    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    // CHECK-NEXT: %8 = p4.get_member_ref(%1) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %9 = p4.get_member_ref(%1) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %10 = p4.load(%9) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%8, %10) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb2

// CHECK-NEXT: ^bb2:  // 2 preds: ^bb0, ^bb1
// CHECK-NEXT: %11 = p4.get_member_ref(%1) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %12 = p4.load(%11) : !p4.ref<i1> -> i1
// CHECK-NEXT: %13 = p4.copy(%12) : i1 -> i1
// CHECK-NEXT: cf.cond_br %13, ^bb3, ^bb4(%13 : i1)

    // CHECK-NEXT: ^bb3:  // pred: ^bb2
    // CHECK-NEXT: %14 = p4.constant false
    // CHECK-NEXT: %15 = p4.copy(%14) : i1 -> i1
    // CHECK-NEXT: cf.br ^bb4(%15 : i1)

// CHECK-NEXT: ^bb4(%16: i1):  // 2 preds: ^bb2, ^bb3
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar() {

// CHECK-NEXT: %0 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %1 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%1, %0) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %2 = p4.get_member_ref(%1) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %3 = p4.load(%2) : !p4.ref<i1> -> i1
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
