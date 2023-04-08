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

def CompareOpKindAttr: I32EnumAttr<
                        "CompareOpKind", "Determines the kind of comparison done by CompareOp",
                        [Equal]> {
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

def FieldDeclOp : P4Op<"field_decl", [Symbol, HasParent<"HeaderOp">]> {
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    let assemblyFormat = "attr-dict $sym_name `:` $type";
}

class P4Dialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<P4Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def HeaderType : P4Dialect_Type<"Header", "header"> {
    let parameters = (ins StringRefParameter<"Header type name">:$name);
    let assemblyFormat = "`<` $name `>`";
}

#endif // STANDALONE_OPS
