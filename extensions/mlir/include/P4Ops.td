#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "P4Dialect.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"

def ConstantOp : P4Op<"constant"> {
    // TODO: for some reason these must be prefixed with Builtin_
    let arguments = (ins Builtin_IntegerAttr:$value);
    let results = (outs Builtin_Integer:$result);

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "mlir::Type":$type, "int64_t":$value), [{
            build($_builder, $_state, type, ::mlir::IntegerAttr::get(type, value));
        }]>
    ];
}

def ControlOp : P4Op<"control", [IsolatedFromAbove, NoTerminator]> {
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
}

def ActionOp : P4Op<"action", [Symbol]> {
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region MinSizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
}

def ApplyOp : P4Op<"apply", [HasParent<"ControlOp">]> {
    let regions = (region MinSizedRegion<1>:$body);
    let assemblyFormat = "attr-dict regions";
}

def ReturnOp : P4Op<"return", [Pure, ParentOneOf<["ActionOp", "ApplyOp"]>, Terminator]> {
    let assemblyFormat = "attr-dict";
}

def CopyOp : P4Op<"copy", [SameOperandsAndResultType]> {
    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict `(` $value `)` `:` type($value) `->` type($result)";
}

def CastOp : P4Op<"cast"> {
    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict `(` $value `)` `:` type($value) `->` type($result)";
}

def Equal: I32EnumAttrCase<"eq", 0>;
def NotEqual: I32EnumAttrCase<"ne", 1>;
def LessThan: I32EnumAttrCase<"lt", 3>;
def LessOrEqual: I32EnumAttrCase<"le", 4>;
def Greater: I32EnumAttrCase<"gt", 5>;
def GreaterOrEqual: I32EnumAttrCase<"ge", 6>;

def CompareOpKindAttr: I32EnumAttr<
                        "CompareOpKind", "Determines the kind of comparison done by CompareOp",
                        [Equal, NotEqual, LessThan, LessOrEqual, Greater, GreaterOrEqual]> {
    let cppNamespace = "p4mlir";
    //let stringToSymbolFnName = "ConvertToEnum";
    //let symbolToStringFnName = "ConvertToString";
}

def CompareOp : P4Op<"cmp", [Pure, SameTypeOperands]> {
    let arguments = (ins CompareOpKindAttr:$kind, Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs I1:$result);
    let assemblyFormat = "attr-dict `(` $lhs `,` $rhs `)` $kind  `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def UninitializedOp : P4Op<"uninitialized"> {
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict `:` type($result)";
}

def HeaderOp : P4Op<"header", [SymbolTable, Symbol, IsolatedFromAbove, NoTerminator]> {
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region MinSizedRegion<1>:$body);
    let assemblyFormat = "attr-dict $sym_name $body";
}

def MemberDeclOp : P4Op<"member_decl", [Symbol]> {
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    let assemblyFormat = "attr-dict $sym_name `:` $type";
}

class ArithmeticOperator<string mnemonic, list<Trait> traits = []>
    : P4Op<mnemonic,
           !listconcat([Pure, SameOperandsAndResultType], traits)> {
    let arguments = (ins Builtin_Integer:$lhs, Builtin_Integer:$rhs);
    let results = (outs Builtin_Integer:$result);
    let assemblyFormat = "attr-dict `(` operands `)` `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def AddOp : ArithmeticOperator<"add"> {}
def SubOp : ArithmeticOperator<"sub"> {}
def MulOp : ArithmeticOperator<"mul"> {}


// TODO: move to separate types file
class P4Dialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def HeaderType : P4Dialect_Type<"Header", "header"> {
    let parameters = (ins StringRefParameter<"Header type name">:$name);
    let assemblyFormat = "`<` $name `>`";
}

def ControlType : P4Dialect_Type<"Control", "control"> {
    let parameters = (ins StringRefParameter<"Control type name">:$name);
    let assemblyFormat = "`<` $name `>`";
}

def RefType : P4Dialect_Type<"Ref", "ref"> {
    let parameters = (ins "mlir::Type":$type);
    let assemblyFormat = "`<` qualified($type) `>`";
}

def AllocOp : P4Op<"alloc"> {
    let results = (outs RefType:$result);
    let assemblyFormat = "attr-dict `:` qualified(type($result)) ";
}

def LoadOp : P4Op<"load"> {
    let arguments = (ins RefType:$addr);
    // TODO: verify result is of type pointed by $addr
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict `(` $addr `)` `:` qualified(type($addr)) `->` qualified(type($result))";
}

def StoreOp : P4Op<"store"> {
    let arguments = (ins RefType:$addr, AnyType:$value);
    // TODO: verify result is of type pointed by $addr
    let assemblyFormat = "attr-dict `(` operands `)` `:` `(` qualified(type(operands)) `)` `->` `(` `)` ";
}

// TODO: the 'member_name' should be a symbol not string
def GetMemberOp : P4Op<"get_member"> {
    let arguments = (ins HeaderType:$base, Builtin_StringAttr:$member_name);
    let results = (outs AnyType:$result);
    let assemblyFormat = "attr-dict `(` $base `)` $member_name `:` qualified(type($base)) `->` type($result)";
}

// TODO: the 'member_name' should be a symbol not string
def GetMemberRefOp : P4Op<"get_member_ref"> {
    let arguments = (ins RefType:$base, Builtin_StringAttr:$member_name);
    let results = (outs RefType:$result);
    let assemblyFormat = "attr-dict `(` $base `)` $member_name `:` qualified(type($base)) `->` qualified(type($result))";
}

// TODO: must have control/parser parent and be within action/apply/state
def SelfOp : P4Op<"self"> {
    let results = (outs RefType:$result);
    let assemblyFormat = "attr-dict `:`  qualified(type($result))";
}

def CallOp : P4Op<"call", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>);
    let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}


#endif // STANDALONE_OPS
