// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

control InnerPipe(bit<10> arg1, in int<16> arg2, out int<16> arg3)(bool flag, int<16> ctr_arg1, MyHeader hdr) {
    apply {}
}

control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4)(int<16> ctr_arg1, MyHeader hdr_arg) {

    InnerPipe(true, ctr_arg1 + 3, hdr_arg) inner1;
    InnerPipe(true, hdr_arg.f1, hdr_arg) inner2;
    // TODO: InnerPipe(true || flag, 3, {2}) inner;

    action bar() {
        int<16> x1;
        return;
    }

    apply {
        bar();
        int<16> x1;
        inner1.apply(1, ctr_arg1, x1);
    }
}

// CHECK: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @InnerPipe(%arg0: ui10, %arg1: si16, %arg2: !p4.ref<si16>)(%arg3: i1, %arg4: si16, %arg5: !p4.header<"MyHeader">) {
// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"InnerPipe">>
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: ui10, %arg1: si16, %arg2: !p4.ref<si16>, %arg3: !p4.ref<si16>)(%arg4: si16, %arg5: !p4.header<"MyHeader">) {

// CHECK-NEXT: p4.member_decl @inner1 : !p4.control<"InnerPipe"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant true
// CHECK-NEXT: %2 = p4.constant 3 : si16
// CHECK-NEXT: %3 = p4.add(%arg4, %2) : (si16, si16) -> si16
// CHECK-NEXT: p4.init !p4.control<"InnerPipe"> (%1, %3, %arg5) : (i1, si16, !p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @inner2 : !p4.control<"InnerPipe"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant true
// CHECK-NEXT: %2 = p4.get_member(%arg5) "f1" : !p4.header<"MyHeader"> -> si16
// CHECK-NEXT: p4.init !p4.control<"InnerPipe"> (%1, %2, %arg5) : (i1, si16, !p4.header<"MyHeader">)
// CHECK-NEXT: }

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
// CHECK-NEXT: %3 = p4.get_member_ref(%0) "inner1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.control<"InnerPipe">>
// CHECK-NEXT: %4 = p4.constant 1 : ui10
// CHECK-NEXT: %5 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.call_apply %3 (%4, %arg4, %5) : (!p4.ref<!p4.control<"InnerPipe">>, ui10, si16, !p4.ref<si16>) -> ()
// CHECK-NEXT: %6 = p4.load(%5) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %6) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }




