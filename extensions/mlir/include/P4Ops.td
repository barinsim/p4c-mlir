#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "P4Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
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

def ActionOp : P4Op<"action"> {
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region MinSizedRegion<1>:$body);
}

def ReturnOp : P4Op<"return", [Pure, HasParent<"ActionOp">, Terminator]> {
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

    //let assemblyFormat = "attr-dict";
}


#endif // STANDALONE_OPS
