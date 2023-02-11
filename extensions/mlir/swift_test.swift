import Foundation

struct MyStruct {
    var a : Int
    var b : Int
    var c : Int
}

func sum(l: Int, r: Int, s: MyStruct) -> (Int, Int) {
    let c = 10
    var o = 5
    @inline(never) func solve(x: Int, out: inout Int) -> Int {
        o = 8
        return l + r + x + s.a + s.b
    }
    solve(x: l, out: &o) 
    return (o, c)
}

let inp = Int(readLine()!)!
let (res1, res2) = sum(l: inp, r: 2, s: MyStruct(a: 7, b: 8, c: 9))
print(res1)



/*
// sum(l:r:s:)
sil hidden [ossa] @$s10swift_test3sum1l1r1sS2i_SiAA8MyStructVtF : $@convention(thin) (Int, Int, MyStruct) -> Int {
// %0 "l"                                         // users: %16, %3
// %1 "r"                                         // users: %16, %4
// %2 "s"                                         // users: %16, %5
bb0(%0 : $Int, %1 : $Int, %2 : $MyStruct):
  debug_value %0 : $Int, let, name "l", argno 1   // id: %3
  debug_value %1 : $Int, let, name "r", argno 2   // id: %4
  debug_value %2 : $MyStruct, let, name "s", argno 3 // id: %5
  %6 = integer_literal $Builtin.IntLiteral, 10    // user: %9
  %7 = metatype $@thin Int.Type                   // user: %9
  // function_ref Int.init(_builtinIntegerLiteral:)
  %8 = function_ref @$sSi22_builtinIntegerLiteralSiBI_tcfC : $@convention(method) (Builtin.IntLiteral, @thin Int.Type) -> Int // user: %9
  %9 = apply %8(%6, %7) : $@convention(method) (Builtin.IntLiteral, @thin Int.Type) -> Int // users: %16, %10
  debug_value %9 : $Int, let, name "c"            // id: %10
  %11 = integer_literal $Builtin.IntLiteral, 4    // user: %14
  %12 = metatype $@thin Int.Type                  // user: %14
  // function_ref Int.init(_builtinIntegerLiteral:)
  %13 = function_ref @$sSi22_builtinIntegerLiteralSiBI_tcfC : $@convention(method) (Builtin.IntLiteral, @thin Int.Type) -> Int // user: %14
  %14 = apply %13(%11, %12) : $@convention(method) (Builtin.IntLiteral, @thin Int.Type) -> Int // user: %16
  // function_ref solve #1 (x:) in sum(l:r:s:)
  %15 = function_ref @$s10swift_test3sum1l1r1sS2i_SiAA8MyStructVtF5solveL_1xS2i_tF : $@convention(thin) (Int, Int, Int, Int, MyStruct) -> Int // user: %16
  %16 = apply %15(%14, %0, %1, %9, %2) : $@convention(thin) (Int, Int, Int, Int, MyStruct) -> Int // user: %17
  return %16 : $Int                               // id: %17
} // end sil function '$s10swift_test3sum1l1r1sS2i_SiAA8MyStructVtF'

// Int.init(_builtinIntegerLiteral:)
sil [transparent] [serialized] @$sSi22_builtinIntegerLiteralSiBI_tcfC : $@convention(method) (Builtin.IntLiteral, @thin Int.Type) -> Int

// solve #1 (x:) in sum(l:r:s:)
sil private [ossa] @$s10swift_test3sum1l1r1sS2i_SiAA8MyStructVtF5solveL_1xS2i_tF : $@convention(thin) (Int, Int, Int, Int, MyStruct) -> Int {
// %0 "x"                                         // users: %18, %5
// %1 "l"                                         // users: %16, %6
// %2 "r"                                         // users: %16, %7
// %3 "c"                                         // users: %20, %8
// %4 "s"                                         // users: %24, %21, %9
bb0(%0 : $Int, %1 : $Int, %2 : $Int, %3 : $Int, %4 : $MyStruct):
  debug_value %0 : $Int, let, name "x", argno 1   // id: %5
  debug_value %1 : $Int, let, name "l", argno 2   // id: %6
  debug_value %2 : $Int, let, name "r", argno 3   // id: %7
  debug_value %3 : $Int, let, name "c", argno 4   // id: %8
  debug_value %4 : $MyStruct, let, name "s", argno 5 // id: %9
  %10 = metatype $@thin Int.Type                  // user: %26
  %11 = metatype $@thin Int.Type                  // user: %23
  %12 = metatype $@thin Int.Type                  // user: %20
  %13 = metatype $@thin Int.Type                  // user: %18
  %14 = metatype $@thin Int.Type                  // user: %16
  // function_ref static Int.+ infix(_:_:)
  %15 = function_ref @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %16
  %16 = apply %15(%1, %2, %14) : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %18
  // function_ref static Int.+ infix(_:_:)
  %17 = function_ref @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %18
  %18 = apply %17(%16, %0, %13) : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %20
  // function_ref static Int.+ infix(_:_:)
  %19 = function_ref @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %20
  %20 = apply %19(%18, %3, %12) : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %23
  %21 = struct_extract %4 : $MyStruct, #MyStruct.a // user: %23
  // function_ref static Int.+ infix(_:_:)
  %22 = function_ref @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %23
  %23 = apply %22(%20, %21, %11) : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %26
  %24 = struct_extract %4 : $MyStruct, #MyStruct.b // user: %26
  // function_ref static Int.+ infix(_:_:)
  %25 = function_ref @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %26
  %26 = apply %25(%23, %24, %10) : $@convention(method) (Int, Int, @thin Int.Type) -> Int // user: %27
  return %26 : $Int                               // id: %27
} // end sil function '$s10swift_test3sum1l1r1sS2i_SiAA8MyStructVtF5solveL_1xS2i_tF'

// static Int.+ infix(_:_:)
sil [transparent] [serialized] @$sSi1poiyS2i_SitFZ : $@convention(method) (Int, Int, @thin Int.Type) -> Int
*/