// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
}

control Pipe(int<16> arg1, in int<16> arg2) {

    const int<16> cst = 42;
    int<16> val1 = 7;
    int<16> val2 = 8;
    MyHeader val3 = {50};

    action foo(inout int<16> arg1, out int<16> arg2) {
        val1 = arg2;
        val3.f1 = val2;
    }

    int<16> val4 = 9;

    action bar() {
        int<16> x1 = 2;
        x1 = arg1;
        if (x1 == 2) {
            val4 = 4;
        } else {
            val4 = val3.f1 + 2;
        }
        x1 = val4;
    }
    apply {
        int<16> x1 = 3;
        foo(x1, val1);
        bar();
    }
}

// CHECK: module {

// CHECK-NEXT: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.control @Pipe(%arg0: si16, %arg1: si16) {

// CHECK-NEXT: p4.member_decl @cst : si16 {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 42 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: p4.init si16 (%2) : (si16)
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @foo(%arg2: !p4.ref<si16>, %arg3: !p4.ref<si16>, %arg4: !p4.ref<!p4.header<"MyHeader">>, %arg5: !p4.ref<si16>, %arg6: !p4.ref<si16>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.load(%arg6) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%arg2, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %2 = p4.get_member_ref(%arg4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %3 = p4.load(%arg3) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %3) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar(%arg2: !p4.ref<si16>, %arg3: !p4.ref<si16>, %arg4: !p4.ref<!p4.header<"MyHeader">>, %arg5: !p4.ref<si16>) {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 2 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.copy(%2) : si16 -> si16
// CHECK-NEXT: %4 = p4.copy(%arg0) : si16 -> si16
// CHECK-NEXT: %5 = p4.constant 2 : si64
// CHECK-NEXT: %6 = p4.cast(%5) : si64 -> si16
// CHECK-NEXT: %7 = p4.cmp(%4, %6) eq : (si16, si16) -> i1

    // CHECK-NEXT: cf.cond_br %7, ^bb1, ^bb2
    // CHECK-NEXT: ^bb1:  // pred: ^bb0
    // CHECK-NEXT: %8 = p4.constant 4 : si16
    // CHECK-NEXT: %9 = p4.cast(%8) : si16 -> si16
    // CHECK-NEXT: p4.store(%arg5, %9) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb3

    // CHECK-NEXT: ^bb2:  // pred: ^bb0
    // CHECK-NEXT: %10 = p4.get_member_ref(%arg4) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
    // CHECK-NEXT: %11 = p4.load(%10) : !p4.ref<si16> -> si16
    // CHECK-NEXT: %12 = p4.constant 2 : si16
    // CHECK-NEXT: %13 = p4.add(%11, %12) : (si16, si16) -> si16
    // CHECK-NEXT: p4.store(%arg5, %13) : (!p4.ref<si16>, si16) -> ()
    // CHECK-NEXT: cf.br ^bb3

// CHECK-NEXT: ^bb3:  // 2 preds: ^bb1, ^bb2
// CHECK-NEXT: %14 = p4.load(%arg5) : !p4.ref<si16> -> si16
// CHECK-NEXT: %15 = p4.copy(%14) : si16 -> si16
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.apply {
// CHECK-NEXT: %0 = p4.self : !p4.ref<!p4.control<"Pipe">>
// CHECK-NEXT: %1 = p4.constant 7 : si16
// CHECK-NEXT: %2 = p4.cast(%1) : si16 -> si16
// CHECK-NEXT: %3 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%3, %2) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %4 = p4.constant 8 : si16
// CHECK-NEXT: %5 = p4.cast(%4) : si16 -> si16
// CHECK-NEXT: %6 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%6, %5) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %7 = p4.constant 50 : si16
// CHECK-NEXT: %8 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %9 = p4.get_member_ref(%8) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%9, %7) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %10 = p4.get_member_ref(%8) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %11 = p4.constant true
// CHECK-NEXT: p4.store(%10, %11) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %12 = p4.load(%8) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %13 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%13, %12) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %14 = p4.constant 9 : si16
// CHECK-NEXT: %15 = p4.cast(%14) : si16 -> si16
// CHECK-NEXT: %16 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%16, %15) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %17 = p4.constant 3 : si16
// CHECK-NEXT: %18 = p4.cast(%17) : si16 -> si16
// CHECK-NEXT: %19 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%19, %18) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %20 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %21 = p4.load(%19) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%20, %21) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %22 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.call @Pipe::@foo(%3, %6, %13, %20, %22) : (!p4.ref<si16>, !p4.ref<si16>, !p4.ref<!p4.header<"MyHeader">>, !p4.ref<si16>, !p4.ref<si16>) -> ()
// CHECK-NEXT: %23 = p4.load(%20) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%19, %23) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %24 = p4.load(%22) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%3, %24) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.call @Pipe::@bar(%3, %6, %13, %16) : (!p4.ref<si16>, !p4.ref<si16>, !p4.ref<!p4.header<"MyHeader">>, !p4.ref<si16>) -> ()
// CHECK-NEXT: p4.return

// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
