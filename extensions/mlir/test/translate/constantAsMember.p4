// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

control InnerPipe(bit<10> arg1, in int<16> arg2, out int<16> arg3)(bool flag, int<16> ctr_arg1, MyHeader hdr) {
    apply {}
}

control Pipe(bit<10> arg1, in int<16> arg2, out int<16> arg3, inout int<16> arg4)(int<16> ctr_arg1, MyHeader hdr_arg) {

    const MyHeader cst1 = hdr_arg;
    const MyHeader cst2 = {2};
    const bool cst3 = false;

    InnerPipe(cst3, cst2.f1 + 3, cst2) inner;

    action bar() {
        int<16> x1;
        return;
    }

    const int<16> cst4 = 3 + cst2.f1;
    const int<16> cst5 = cst4;
    const int<16> cst6 = 7;
    const int<16> cst7 = ctr_arg1;
    const int<16> cst8 = ctr_arg1 + ctr_arg1 + 10;

    apply {
        bar();
        int<16> x1 = cst7;
        if (x1 == cst8) {
            x1 = cst5;
        } else {
            inner.apply(1, cst2.f1, x1);
            x1 = 3 + cst1.f1;
        }
        bool val = cst2.isValid();
        val = hdr_arg.isValid();
    }
}

// CHECK: module {
// CHECK-NEXT: p4.header @MyHeader {
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
// CHECK-NEXT: p4.member_decl @cst1 : !p4.header<"MyHeader"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: p4.init !p4.header<"MyHeader"> (%arg5) : (!p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst2 : !p4.header<"MyHeader"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 2 : si16
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %3 = p4.get_member_ref(%2) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%3, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %4 = p4.get_member_ref(%2) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %5 = p4.constant true
// CHECK-NEXT: p4.store(%4, %5) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %6 = p4.load(%2) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.init !p4.header<"MyHeader"> (%6) : (!p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst3 : i1 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant false
// CHECK-NEXT: p4.init i1 (%1) : (i1)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @inner : !p4.control<"InnerPipe"> {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst3" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<i1>
// CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<i1> -> i1
// CHECK-NEXT: %3 = p4.get_member_ref(%0) "cst2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %4 = p4.get_member_ref(%3) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %5 = p4.load(%4) : !p4.ref<si16> -> si16
// CHECK-NEXT: %6 = p4.constant 3 : si16
// CHECK-NEXT: %7 = p4.add(%5, %6) : (si16, si16) -> si16
// CHECK-NEXT: %8 = p4.get_member_ref(%0) "cst2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %9 = p4.load(%8) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: p4.init !p4.control<"InnerPipe"> (%2, %7, %9) : (i1, si16, !p4.header<"MyHeader">)
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar() {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst4 : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 3 : si16
// CHECK-NEXT: %2 = p4.get_member_ref(%0) "cst2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %3 = p4.get_member_ref(%2) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %4 = p4.load(%3) : !p4.ref<si16> -> si16
// CHECK-NEXT: %5 = p4.add(%1, %4) : (si16, si16) -> si16
// CHECK-NEXT: p4.init si16 (%5) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst5 : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst4" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
// CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.init si16 (%2) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst6 : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 7 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: p4.init si16 (%2) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst7 : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: p4.init si16 (%arg4) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.member_decl @cst8 : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.add(%arg4, %arg4) : (si16, si16) -> si16
// CHECK-NEXT: %2 = p4.constant 10 : si16
// CHECK-NEXT: %3 = p4.add(%1, %2) : (si16, si16) -> si16
// CHECK-NEXT: p4.init si16 (%3) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: p4.call @Pipe::@bar() : () -> ()
// CHECK-NEXT: %1 = p4.get_member_ref(%0) "cst7" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
// CHECK-NEXT: %2 = p4.load(%1) : !p4.ref<si16> -> si16
// CHECK-NEXT: %3 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%3, %2) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %4 = p4.load(%3) : !p4.ref<si16> -> si16
// CHECK-NEXT: %5 = p4.get_member_ref(%0) "cst8" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
// CHECK-NEXT: %6 = p4.load(%5) : !p4.ref<si16> -> si16
// CHECK-NEXT: %7 = p4.cmp(%4, %6) eq : (si16, si16) -> i1
// CHECK-NEXT: cf.cond_br %7, ^bb1, ^bb2

    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    // CHECK-NEXT: %8 = p4.get_member_ref(%0) "cst5" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<si16>
    // CHECK-NEXT: %9 = p4.load(%8) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%3, %9) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb3

    // CHECK-NEXT: ^bb2:  // pred: ^bb0
    // CHECK-NEXT: %10 = p4.get_member_ref(%0) "inner" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.control<"InnerPipe">>
    // CHECK-NEXT: %11 = p4.constant 1 : ui10
    // CHECK-NEXT: %12 = p4.get_member_ref(%0) "cst2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: %13 = p4.get_member_ref(%12) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %14 = p4.load(%13) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %15 = p4.alloc : !p4.ref<si16>
    // CHECK-NEXT: p4.call_apply %10 (%11, %14, %15) : (!p4.ref<!p4.control<"InnerPipe">>, ui10, si16, !p4.ref<si16>) -> ()
    // CHECK-NEXT: %16 = p4.load(%15) : !p4.ref<si16> -> si16
    // CHECK-NEXT: p4.store(%3, %16) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: %17 = p4.constant 3 : si16
    // CHECK-NEXT: %18 = p4.get_member_ref(%0) "cst1" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
    // CHECK-NEXT: %19 = p4.get_member_ref(%18) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %20 = p4.load(%19) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %21 = p4.add(%17, %20) : (si16, si16) -> si16
    // CHECK-NEXT: p4.store(%3, %21) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb3

// CHECK-NEXT: ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: %22 = p4.get_member_ref(%0) "cst2" : !p4.ref<!p4.control<"Pipe">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %23 = p4.get_member_ref(%22) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %24 = p4.load(%23) : !p4.ref<i1> -> i1
// CHECK-NEXT: %25 = p4.copy(%24) : i1 -> i1
// CHECK-NEXT: %26 = p4.get_member(%arg5) "__valid" : !p4.header<"MyHeader"> -> i1
// CHECK-NEXT: %27 = p4.copy(%26) : i1 -> i1
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: }
// CHECK-NEXT: }
