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
// CHECK-NEXT: %0 = p4.tuple() : () -> !p4.struct<"EmptyStruct">
// CHECK-NEXT: %1 = p4.copy(%0) : !p4.struct<"EmptyStruct"> -> !p4.struct<"EmptyStruct">
// CHECK-NEXT: %2 = p4.constant 1 : si10
// CHECK-NEXT: %3 = p4.constant 2 : si16
// CHECK-NEXT: %4 = p4.constant 3 : ui16
// CHECK-NEXT: %5 = p4.constant 4 : ui10
// CHECK-NEXT: %6 = p4.constant true
// CHECK-NEXT: %7 = p4.tuple(%3, %4, %5, %6) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %8 = p4.constant 5 : si16
// CHECK-NEXT: %9 = p4.constant 6 : ui16
// CHECK-NEXT: %10 = p4.constant 7 : ui10
// CHECK-NEXT: %11 = p4.constant true
// CHECK-NEXT: %12 = p4.tuple(%8, %9, %10, %11) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %13 = p4.tuple(%2, %7, %12) : (si10, !p4.header<"MyHeader">, !p4.header<"MyHeader">) -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %14 = p4.copy(%13) : !p4.struct<"MyInnerInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %15 = p4.uninitialized : !p4.header<"MyHeader">
// CHECK-NEXT: %16 = p4.constant 1 : si10
// CHECK-NEXT: %17 = p4.constant 2 : si16
// CHECK-NEXT: %18 = p4.constant 3 : ui16
// CHECK-NEXT: %19 = p4.get_member(%15) "f3" : !p4.header<"MyHeader"> -> ui10
// CHECK-NEXT: %20 = p4.constant true
// CHECK-NEXT: %21 = p4.tuple(%17, %18, %19, %20) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %22 = p4.tuple(%16, %21, %15) : (si10, !p4.header<"MyHeader">, !p4.header<"MyHeader">) -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %23 = p4.copy(%22) : !p4.struct<"MyInnerInnerStruct"> -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %24 = p4.constant 1 : si10
// CHECK-NEXT: %25 = p4.constant 2 : si16
// CHECK-NEXT: %26 = p4.constant 3 : ui16
// CHECK-NEXT: %27 = p4.constant 4 : ui10
// CHECK-NEXT: %28 = p4.constant true
// CHECK-NEXT: %29 = p4.tuple(%25, %26, %27, %28) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %30 = p4.constant 1 : si10
// CHECK-NEXT: %31 = p4.constant 2 : si16
// CHECK-NEXT: %32 = p4.constant 3 : ui16
// CHECK-NEXT: %33 = p4.constant 4 : ui10
// CHECK-NEXT: %34 = p4.constant true
// CHECK-NEXT: %35 = p4.tuple(%31, %32, %33, %34) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %36 = p4.constant 5 : si16
// CHECK-NEXT: %37 = p4.constant 6 : ui16
// CHECK-NEXT: %38 = p4.constant 7 : ui10
// CHECK-NEXT: %39 = p4.constant true
// CHECK-NEXT: %40 = p4.tuple(%36, %37, %38, %39) : (si16, ui16, ui10, i1) -> !p4.header<"MyHeader">
// CHECK-NEXT: %41 = p4.tuple(%30, %35, %40) : (si10, !p4.header<"MyHeader">, !p4.header<"MyHeader">) -> !p4.struct<"MyInnerInnerStruct">
// CHECK-NEXT: %42 = p4.tuple(%24, %29, %41) : (si10, !p4.header<"MyHeader">, !p4.struct<"MyInnerInnerStruct">) -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: %43 = p4.copy(%42) : !p4.struct<"MyInnerStruct"> -> !p4.struct<"MyInnerStruct">
// CHECK-NEXT: %44 = p4.get_member(%43) "f2" : !p4.struct<"MyInnerStruct"> -> !p4.header<"MyHeader">
// CHECK-NEXT: %45 = p4.get_member(%44) "f3" : !p4.header<"MyHeader"> -> ui10
// CHECK-NEXT: %46 = p4.tuple(%43, %43, %45) : (!p4.struct<"MyInnerStruct">, !p4.struct<"MyInnerStruct">, ui10) -> !p4.struct<"MyStruct">
// CHECK-NEXT: %47 = p4.alloc : !p4.ref<!p4.struct<"MyStruct">>
// CHECK-NEXT: p4.store(%47, %46) : (!p4.ref<!p4.struct<"MyStruct">>, !p4.struct<"MyStruct">) -> ()
// CHECK-NEXT: %48 = p4.get_member_ref(%47) "f3" : !p4.ref<!p4.struct<"MyStruct">> -> !p4.ref<ui10>
// CHECK-NEXT: %49 = p4.constant 42 : ui10
// CHECK-NEXT: %50 = p4.cast(%49) : ui10 -> ui10
// CHECK-NEXT: p4.store(%48, %50) : (!p4.ref<ui10>, ui10) -> ()
// CHECK-NEXT: p4.return
// CHECK-NEXT: }
