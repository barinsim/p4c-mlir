// RUN: p4c-mlir-translate %s | FileCheck %s

header MyHeader {
    int<16> f1;
    bit<16> f2;
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

action foo() {
    EmptyStruct str_empty = {};

    MyInnerInnerStruct str1 = {1, {2, 3, 4}, {5, 6, 7}};

    MyHeader hdr;
    MyInnerInnerStruct str2 = {1, {2, 3, hdr.f3}, hdr};

    MyInnerStruct str3 = {1, {2, 3, 4}, {1, {2, 3, 4}, {5, 6, 7}}};

    MyStruct str4 = {str3, str3, str3.f2.f3};
    str4.f3 = 42;
}

// CHECK: p4.action @foo() {
// CHECK-NEXT: %0 = p4.alloc : !p4.ref<!p4.struct<"EmptyStruct">>
// CHECK-NEXT: %1 = p4.load(%0) : !p4.ref<!p4.struct<"EmptyStruct">> -> !p4.struct<"EmptyStruct">
// CHECK-NEXT: %2 = p4.copy(%1) : !p4.struct<"EmptyStruct"> -> !p4.struct<"EmptyStruct">
// CHECK-NEXT: %3 = p4.constant 1 : si10
// CHECK-NEXT: %4 = p4.constant 2 : si16
// CHECK-NEXT: %5 = p4.constant 3 : ui16
// CHECK-NEXT: %6 = p4.constant 4 : ui10
// CHECK-NEXT: %7 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %8 = p4.get_member_ref(%7) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%8, %4) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %9 = p4.get_member_ref(%7) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%9, %5) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %10 = p4.get_member_ref(%7) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%10, %6) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %11 = p4.get_member_ref(%7) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %12 = p4.constant true
// CHECK-NEXT: p4.store(%11, %12) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %13 = p4.load(%7) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %14 = p4.constant 5 : si16
// CHECK-NEXT: %15 = p4.constant 6 : ui16
// CHECK-NEXT: %16 = p4.constant 7 : ui10
// CHECK-NEXT: %17 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %18 = p4.get_member_ref(%17) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%18, %14) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %19 = p4.get_member_ref(%17) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%19, %15) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %20 = p4.get_member_ref(%17) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%20, %16) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %21 = p4.get_member_ref(%17) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %22 = p4.constant true
// CHECK-NEXT: p4.store(%21, %22) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %23 = p4.load(%17) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %24 = p4.alloc : !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %25 = p4.get_member_ref(%24) "f1" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: p4.store(%25, %3) : (!p4.ref<si10>, si10) -> ()
// CHECK-NEXT: %26 = p4.get_member_ref(%24) "f2" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%26, %13) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %27 = p4.get_member_ref(%24) "f3" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%27, %23) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %28 = p4.load(%24) : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %29 = p4.copy(%28) : !p4.struct<"MyInnerInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %30 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %31 = p4.constant 1 : si10
// CHECK-NEXT: %32 = p4.constant 2 : si16
// CHECK-NEXT: %33 = p4.constant 3 : ui16
// CHECK-NEXT: %34 = p4.get_member(%30) "f3" : !p4.header<"MyHeader"> -> ui10
// CHECK-NEXT: %35 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %36 = p4.get_member_ref(%35) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%36, %32) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %37 = p4.get_member_ref(%35) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%37, %33) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %38 = p4.get_member_ref(%35) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%38, %34) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %39 = p4.get_member_ref(%35) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %40 = p4.constant true
// CHECK-NEXT: p4.store(%39, %40) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %41 = p4.load(%35) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %42 = p4.alloc : !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %43 = p4.get_member_ref(%42) "f1" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: p4.store(%43, %31) : (!p4.ref<si10>, si10) -> ()
// CHECK-NEXT: %44 = p4.get_member_ref(%42) "f2" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%44, %41) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %45 = p4.get_member_ref(%42) "f3" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%45, %30) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %46 = p4.load(%42) : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %47 = p4.copy(%46) : !p4.struct<"MyInnerInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %48 = p4.constant 1 : si10
// CHECK-NEXT: %49 = p4.constant 2 : si16
// CHECK-NEXT: %50 = p4.constant 3 : ui16
// CHECK-NEXT: %51 = p4.constant 4 : ui10
// CHECK-NEXT: %52 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %53 = p4.get_member_ref(%52) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%53, %49) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %54 = p4.get_member_ref(%52) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%54, %50) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %55 = p4.get_member_ref(%52) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%55, %51) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %56 = p4.get_member_ref(%52) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %57 = p4.constant true
// CHECK-NEXT: p4.store(%56, %57) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %58 = p4.load(%52) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %59 = p4.constant 1 : si10
// CHECK-NEXT: %60 = p4.constant 2 : si16
// CHECK-NEXT: %61 = p4.constant 3 : ui16
// CHECK-NEXT: %62 = p4.constant 4 : ui10
// CHECK-NEXT: %63 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %64 = p4.get_member_ref(%63) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%64, %60) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %65 = p4.get_member_ref(%63) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%65, %61) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %66 = p4.get_member_ref(%63) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%66, %62) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %67 = p4.get_member_ref(%63) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %68 = p4.constant true
// CHECK-NEXT: p4.store(%67, %68) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %69 = p4.load(%63) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %70 = p4.constant 5 : si16
// CHECK-NEXT: %71 = p4.constant 6 : ui16
// CHECK-NEXT: %72 = p4.constant 7 : ui10
// CHECK-NEXT: %73 = p4.alloc : !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: %74 = p4.get_member_ref(%73) "f1" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<si16>
// CHECK-NEXT: p4.store(%74, %70) : (!p4.ref<si16>, si16) -> ()
// CHECK-NEXT: %75 = p4.get_member_ref(%73) "f2" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui16>
// CHECK-NEXT: p4.store(%75, %71) : (!p4.ref<ui16>, ui16) -> ()
// CHECK-NEXT: %76 = p4.get_member_ref(%73) "f3" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%76, %72) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %77 = p4.get_member_ref(%73) "__valid" : !p4.ref<!p4.header<"MyHeader">> -> !p4.ref<i1>
// CHECK-NEXT: %78 = p4.constant true
// CHECK-NEXT: p4.store(%77, %78) : (!p4.ref<i1>, i1) -> ()
// CHECK-NEXT: %79 = p4.load(%73) : !p4.ref<!p4.header<"MyHeader">> -> !p4.header<"MyHeader">
// CHECK-NEXT: %80 = p4.alloc : !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: %81 = p4.get_member_ref(%80) "f1" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: p4.store(%81, %59) : (!p4.ref<si10>, si10) -> ()
// CHECK-NEXT: %82 = p4.get_member_ref(%80) "f2" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%82, %69) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %83 = p4.get_member_ref(%80) "f3" : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%83, %79) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %84 = p4.load(%80) : !p4.ref<!p4.struct<"MyInnerInnerStruct">> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %85 = p4.alloc : !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: %86 = p4.get_member_ref(%85) "f1" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<si10>
// CHECK-NEXT: p4.store(%86, %48) : (!p4.ref<si10>, si10) -> ()
// CHECK-NEXT: %87 = p4.get_member_ref(%85) "f2" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.header<"MyHeader">>
// CHECK-NEXT: p4.store(%87, %58) : (!p4.ref<!p4.header<"MyHeader">>, !p4.header<"MyHeader">) -> ()
// CHECK-NEXT: %88 = p4.get_member_ref(%85) "f3" : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.ref<!p4.struct<"MyInnerInnerStruct">>
// CHECK-NEXT: p4.store(%88, %84) : (!p4.ref<!p4.struct<"MyInnerInnerStruct">>, !p4.struct<"MyInnerInnerStruct">) -> ()
// CHECK-NEXT: %89 = p4.load(%85) : !p4.ref<!p4.struct<"MyInnerStruct">> -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: %90 = p4.copy(%89) : !p4.struct<"MyInnerStruct"> -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: %91 = p4.get_member(%90) "f2" : !p4.struct<"MyInnerStruct"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %92 = p4.get_member(%91) "f3" : !p4.header<"MyHeader"> -> ui10
// CHECK-NEXT: %93 = p4.alloc : !p4.ref<!p4.struct<"MyStruct">>
// CHECK-NEXT: %94 = p4.get_member_ref(%93) "f1" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: p4.store(%94, %90) : (!p4.ref<!p4.struct<"MyInnerStruct">>, !p4.struct<"MyInnerStruct">) -> ()
// CHECK-NEXT: %95 = p4.get_member_ref(%93) "f2" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<!p4.struct<"MyInnerStruct">>
// CHECK-NEXT: p4.store(%95, %90) : (!p4.ref<!p4.struct<"MyInnerStruct">>, !p4.struct<"MyInnerStruct">) -> ()
// CHECK-NEXT: %96 = p4.get_member_ref(%93) "f3" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<ui10>
// CHECK-NEXT: p4.store(%96, %92) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: %97 = p4.load(%93) : !p4.ref<!p4.struct<"MyStruct">> -> !p4.struct<"MyStruct">
// CHECK-NEXT: %98 = p4.alloc : !p4.ref<!p4.struct<"MyStruct">>
// CHECK-NEXT: p4.store(%98, %97) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.struct<"MyStruct">) -> ()
// CHECK-NEXT: %99 = p4.get_member_ref(%98) "f3" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<ui10>
// CHECK-NEXT: %100 = p4.constant 42 : ui10
// CHECK-NEXT: %101 = p4.cast(%100) : ui10 -> ui10
// CHECK-NEXT: p4.store(%99, %101) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
