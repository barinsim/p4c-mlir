// RUN: p4c-mlir-translate %s | FileCheck %s

control InnerPipe(bit<10> arg1, in int<16> arg2, out int<16> arg3) {
    apply {}
}

control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4) {

    InnerPipe() inner;

    action bar() {
        int<16> x1;
        return;
    }

    apply {
        bar();
        int<16> x1;
        inner.apply(1, 2, x1);
    }
}

// CHECK: p4.control @InnerPipe(%arg0: ui10, %arg1: si16, %arg2: !p4.ref<si16>) {

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"InnerPipe">>
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: ui10, %arg1: si16, %arg2: !p4.ref<si16>, %arg3: !p4.ref<si16>) {

// CHECK-NEXT: p4.member_decl @inner : !p4.control<"InnerPipe">

// CHECK-NEXT: p4.action @bar() {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: p4.call @Pipe::@bar() : () -> ()
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%2, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %3 = p4.get_member_ref(%0) "inner" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.control<"InnerPipe">>
// CHECK-NEXT: %4 = p4.constant 1 : ui10
// CHECK-NEXT: %5 = p4.constant 2 : si64
// CHECK-NEXT: %6 = p4.cast(%5) : si64 -> si16
// CHECK-NEXT: %7 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.call_apply %3 (%4, %6, %7) : (!p4.ref<!p4.control<"InnerPipe">>, ui10, si16, !p4.ref<si16>) -> ()
// CHECK-NEXT: %8 = p4.load(%7) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %8) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }


