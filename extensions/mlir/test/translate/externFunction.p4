// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    int<16> f2;
    int<16> f3;
}

extern int<16> bar(inout int<16> arg1, in int<16> arg2, in MyHeader arg3);

extern void baz();

extern void foo<M, N>(M arg1, N arg2);
extern bit<2> foo<M>(in M arg1);
extern M foo<M, N>(out M arg1, inout N arg2, in N arg3);

action foo() {
    MyHeader hdr;

    int<16> x1;
    int<16> x2;
    bar(x1, x2, hdr);

    int<16> x3 = bar(x1, x2, hdr);

    bar(x1, bar(x1, x2, hdr), hdr);

    baz();

    foo<int<32>, bit<10>>(3, 2);
    foo<int<32>>(5);
    // TODO: foo<MyHeader>({1, 2, 3});
    foo<MyHeader>(hdr);

    int<40> x4 = 6;
    foo<int<40>, int<40>>(x4, x4, 9);
}

// CHECK: p4.extern @bar_3(!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: p4.extern @baz_0()

// CHECK-NEXT: p4.extern @foo_2<@M, @N>(!p4.type_var<@M>, !p4.type_var<@N>)
// CHECK-NEXT: p4.extern @foo_1<@M>(!p4.type_var<@M>) -> ui2
// CHECK-NEXT: p4.extern @foo_3<@M, @N>(!p4.ref<!p4.type_var<@M>>, !p4.ref<!p4.type_var<@N>>, !p4.type_var<@N>) -> !p4.type_var<@M>

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%2, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %3 = p4.uninitialized : si16
// CHECK-NEXT: %4 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %5 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%4, %5) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %6 = p4.call @bar_3(%4, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %7 = p4.load(%4) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %7) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %8 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %9 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%8, %9) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %10 = p4.call @bar_3(%8, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %11 = p4.load(%8) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %11) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %12 = p4.copy(%10) : si16 -> si16
// CHECK-NEXT: %13 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %14 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%13, %14) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %15 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %16 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%15, %16) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %17 = p4.call @bar_3(%15, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %18 = p4.load(%15) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %18) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %19 = p4.call @bar_3(%13, %17, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %20 = p4.load(%13) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %20) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.call @baz_0() : () -> ()
// CHECK-NEXT: %21 = p4.constant 3 : si32
// CHECK-NEXT: %22 = p4.constant 2 : ui10
// CHECK-NEXT: p4.call @foo_2<si32, ui10>(%21, %22) : (si32, ui10) -> ()
// CHECK-NEXT: %23 = p4.constant 5 : si64
// CHECK-NEXT: %24 = p4.cast(%23) : si64 -> si32
// CHECK-NEXT: %25 = p4.call @foo_1<si32>(%24) : (si32) -> ui2
// CHECK-NEXT: %26 = p4.call @foo_1<!p4.header<"MyHeader">>(%0) : (!p4.header<"MyHeader">) -> ui2
// CHECK-NEXT: %27 = p4.constant 6 : si40
// CHECK-NEXT: %28 = p4.cast(%27) : si40 -> si40
// CHECK-NEXT: %29 = p4.alloc : !p4.ref<si40>
// CHECK-NEXT: p4.store(%29, %28) : (!p4.ref<si40>, si40) -> ()
// CHECK-NEXT: %30 = p4.alloc : !p4.ref<si40>
// CHECK-NEXT: %31 = p4.alloc : !p4.ref<si40>
// CHECK-NEXT: %32 = p4.load(%29) : !p4.ref<si40> -> si40
// CHECK-NEXT: p4.store(%31, %32) : (!p4.ref<si40>, si40) -> ()
// CHECK-NEXT: %33 = p4.constant 9 : si64
// CHECK-NEXT: %34 = p4.cast(%33) : si64 -> si40
// CHECK-NEXT: %35 = p4.call @foo_3<si40, si40>(%30, %31, %34) : (!p4.ref<si40>, !p4.ref<si40>, si40) -> !p4.type_var<@M>
// CHECK-NEXT: %36 = p4.load(%30) : !p4.ref<si40> -> si40
// CHECK-NEXT: p4.store(%29, %36) : (!p4.ref<si40>, si40) -> ()
// CHECK-NEXT: %37 = p4.load(%31) : !p4.ref<si40> -> si40
// CHECK-NEXT: p4.store(%29, %37) : (!p4.ref<si40>, si40) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
