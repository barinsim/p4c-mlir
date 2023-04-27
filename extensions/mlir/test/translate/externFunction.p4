// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    int<16> f2;
    int<16> f3;
}

extern int<16> bar(inout int<16> arg1, in int<16> arg2, in MyHeader arg3);

extern void baz();

action foo() {
    MyHeader hdr;

    int<16> x1;
    int<16> x2;
    bar(x1, x2, hdr);

    int<16> x3 = bar(x1, x2, hdr);

    bar(x1, bar(x1, x2, hdr), hdr);

    baz();
}

// CHECK: p4.extern @bar(!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16

// CHECK-NEXT: p4.extern @baz()

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %1 = p4.uninitialized : si16
// CHECK-NEXT: %2 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: p4.store(%2, %1) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %3 = p4.uninitialized : si16
// CHECK-NEXT: %4 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %5 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%4, %5) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %6 = p4.call @bar(%4, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %7 = p4.load(%4) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %7) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %8 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %9 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%8, %9) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %10 = p4.call @bar(%8, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %11 = p4.load(%8) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %11) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %12 = p4.copy(%10) : si16 -> si16
// CHECK-NEXT: %13 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %14 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%13, %14) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %15 = p4.alloc : !p4.ref<si16>
// CHECK-NEXT: %16 = p4.load(%2) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%15, %16) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %17 = p4.call @bar(%15, %3, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %18 = p4.load(%15) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %18) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %19 = p4.call @bar(%13, %17, %0) : (!p4.ref<si16>, si16, !p4.header<"MyHeader">) -> si16
// CHECK-NEXT: %20 = p4.load(%13) : !p4.ref<si16> -> si16
// CHECK-NEXT: p4.store(%2, %20) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: p4.call @baz() : () -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
