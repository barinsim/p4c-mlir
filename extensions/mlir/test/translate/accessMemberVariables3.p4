// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    int<16> f2;
    bit<10> f3;
}

struct MyInnerInnerStruct {
    int<10> f1;
    MyHeader f2;
    MyHeader f3;
}

struct MyInnerStruct {
    int<10> f1;
    MyHeader f2;
    MyInnerInnerStruct f3;
}

struct MyStruct {
    MyInnerStruct f1;
    MyInnerStruct f2;
    bit<10> f3;
}

struct EmptyStruct {}

action bar(inout MyStruct arg1, out MyInnerInnerStruct arg2, in MyStruct arg3, MyInnerStruct arg4) {
    return;
}

action foo() {
    MyStruct str_read_only;
    bit<10> tmp;
    tmp = str_read_only.f3;
    tmp = str_read_only.f1.f3.f2.f3;

    MyStruct str;

    str.f3 = 2;

    bit<10> x1;
    x1 = str.f3;

    int<10> x2;
    x2 = str.f1.f1;
    str.f1.f1 = x2;

    MyHeader hdr1 = str.f2.f2;

    MyInnerInnerStruct hdr2 = str.f2.f3;

    x1 = str.f1.f2.f3;
    str.f1.f2.f3 = 5;

    str.f1.f3.f3.f1 = str.f1.f3.f3.f2 + 42;

    bar(str, str.f1.f3, str, str.f2);
}

// CHECK: p4.header @MyHeader {
// CHECK-NEXT: p4.member_decl @f1 : si16
// CHECK-NEXT: p4.member_decl @f2 : si16
// CHECK-NEXT: p4.member_decl @f3 : ui10
// CHECK-NEXT: p4.valid_bit_decl @__valid : i1
// CHECK-NEXT: }

// CHECK-NEXT: p4.struct @MyInnerInnerStruct {
// CHECK-NEXT: p4.member_decl @f1 : si10
// CHECK-NEXT: p4.member_decl @f2 : !p4.header<"MyHeader">
// CHECK-NEXT: p4.member_decl @f3 : !p4.header<"MyHeader">
// CHECK-NEXT: }

// CHECK-NEXT: p4.struct @MyInnerStruct {
// CHECK-NEXT: p4.member_decl @f1 : si10
// CHECK-NEXT: p4.member_decl @f2 : !p4.header<"MyHeader">
// CHECK-NEXT: p4.member_decl @f3 : !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: }

// CHECK-NEXT: p4.struct @MyStruct {
// CHECK-NEXT: p4.member_decl @f1 : !p4.struct<"MyInnerStruct">
// CHECK-NEXT: p4.member_decl @f2 : !p4.struct<"MyInnerStruct">
// CHECK-NEXT: p4.member_decl @f3 : ui10
// CHECK-NEXT: }

// CHECK-NEXT: p4.struct @EmptyStruct {
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @bar(%arg0: !p4.ref<!p4.struct<"MyStruct">>, %arg1: !p4.ref<!p4.struct<"MyInnerInnerStruct">>, %arg2: !p4.struct<"MyStruct">, %arg3: !p4.struct<"MyInnerStruct">) {
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

// CHECK-NEXT: p4.action @foo() {
// CHECK-NEXT: %0 = p4.uninitialized : !p4.struct<"MyStruct">
// CHECK-NEXT: %1 = p4.uninitialized : ui10
// CHECK-NEXT: %2 = p4.get_member(%0) "f3" : !p4.struct<"MyStruct"> -> ui10
// CHECK-NEXT: %3 = p4.copy(%2) : ui10 -> ui10
// CHECK-NEXT: %4 = p4.get_member(%0) "f1" : !p4.struct<"MyStruct"> -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: %5 = p4.get_member(%4) "f3" : !p4.struct<"MyInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %6 = p4.get_member(%5) "f2" : !p4.struct<"MyInnerInnerStruct"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %7 = p4.get_member(%6) "f3" : !p4.header<"MyHeader"> -> ui10
// CHECK-NEXT: %8 = p4.copy(%7) : ui10 -> ui10
// CHECK-NEXT: %9 = p4.uninitialized : !p4.struct<"MyStruct">
// CHECK-NEXT: %10 = p4.alloc : !p4.ref<!p4.struct<"MyStruct">>
// CHECK-NEXT: p4.store(%10, %9) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.struct<"MyStruct">) -> ()
// CHECK-NEXT: %11 = p4.get_member_ref(%10) "f3" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<ui10>
// CHECK-NEXT: %12 = p4.constant 2 : ui10
// CHECK-NEXT: %13 = p4.cast(%12) : ui10 -> ui10
// CHECK-NEXT: p4.store(%11, %13) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %14 = p4.uninitialized : ui10
// CHECK-NEXT: %15 = p4.get_member_ref(%10) "f3" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<ui10>
// CHECK-NEXT: %16 = p4.load(%15) : !p4.ref<ui10> -> ui10
// CHECK-NEXT: %17 = p4.copy(%16) : ui10 -> ui10
// CHECK-NEXT: %18 = p4.uninitialized : si10
// CHECK-NEXT: %19 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %20 = p4.get_member_ref(%19) "f1" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: %21 = p4.load(%20) : !p4.ref<si10> -> si10
// CHECK-NEXT: %22 = p4.copy(%21) : si10 -> si10
// CHECK-NEXT: %23 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %24 = p4.get_member_ref(%23) "f1" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: p4.store(%24, %22) : (!p4.ref<si10>, si10) -> ()
// CHECK-NEXT: %25 = p4.get_member_ref(%10) "f2" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %26 = p4.get_member_ref(%25) "f2" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %27 = p4.load(%26) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %28 = p4.copy(%27) : !p4.header<"MyHeader"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %29 = p4.get_member_ref(%10) "f2" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %30 = p4.get_member_ref(%29) "f3" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %31 = p4.load(%30) : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %32 = p4.copy(%31) : !p4.struct<"MyInnerInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %33 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %34 = p4.get_member_ref(%33) "f2" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %35 = p4.get_member_ref(%34) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: %36 = p4.load(%35) : !p4.ref<ui10> -> ui10
// CHECK-NEXT: %37 = p4.copy(%36) : ui10 -> ui10
// CHECK-NEXT: %38 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %39 = p4.get_member_ref(%38) "f2" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %40 = p4.get_member_ref(%39) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: %41 = p4.constant 5 : ui10
// CHECK-NEXT: %42 = p4.cast(%41) : ui10 -> ui10
// CHECK-NEXT: p4.store(%40, %42) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %43 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %44 = p4.get_member_ref(%43) "f3" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %45 = p4.get_member_ref(%44) "f3" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %46 = p4.get_member_ref(%45) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %47 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %48 = p4.get_member_ref(%47) "f3" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %49 = p4.get_member_ref(%48) "f3" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %50 = p4.get_member_ref(%49) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: %51 = p4.load(%50) : !p4.ref<si16> -> si16
// CHECK-NEXT: %52 = p4.constant 42 : si16
// CHECK-NEXT: %53 = p4.add(%51, %52) : (si16, si16) -> si16
// CHECK-NEXT: p4.store(%46, %53) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %54 = p4.alloc : !p4.ref<!p4.struct<"MyStruct">>
// CHECK-NEXT: %55 = p4.load(%10) : !p4.ref<!p4.struct<"MyStruct">> -> !p4.struct<"MyStruct">
// CHECK-NEXT: p4.store(%54, %55) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.struct<"MyStruct">) -> ()
// CHECK-NEXT: %56 = p4.get_member_ref(%10) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %57 = p4.get_member_ref(%56) "f3" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %58 = p4.alloc : !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %59 = p4.load(%10) : !p4.ref<!p4.struct<"MyStruct">> -> !p4.struct<"MyStruct">
// CHECK-NEXT: %60 = p4.get_member_ref(%10) "f2" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %61 = p4.load(%60) : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: p4.call @bar(%54, %58, %59, %61) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.ref<!p4.struct<"MyInnerInnerStruct">>, !p4.struct<"MyStruct">, !p4.struct<"MyInnerStruct">) -> ()
// CHECK-NEXT: %62 = p4.load(%54) : !p4.ref<!p4.struct<"MyStruct">> -> !p4.struct<"MyStruct">
// CHECK-NEXT: p4.store(%10, %62) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.struct<"MyStruct">) -> ()
// CHECK-NEXT: %63 = p4.load(%58) : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: p4.store(%57, %63) : (!p4.ref<!p4.struct<"MyInnerInnerStruct">>, !p4.struct<"MyInnerInnerStruct">) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }

