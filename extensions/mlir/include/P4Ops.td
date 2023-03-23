#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "P4Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"


def ConstantOp : P4Op<"constant"> {
    // TODO: for some reason these must be prefixed with Builtin_
    let arguments = (ins Builtin_IntegerAttr:$value);
    let results = (outs Builtin_Integer:$result);
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "mlir::Type":$type, "int64_t":$value), [{
            build($_builder, $_state, type, ::mlir::IntegerAttr::get(type, value));
        }]>
    ];
}


def ActionOp : P4Op<"action"> {
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region MinSizedRegion<1>:$body);

  // let builders = [OpBuilder<(ins
  //   "StringRef":$name, "FunctionType":$type,
  //   CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
  //   CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  // >];
}


def ReturnOp : P4Op<"return", [Pure, HasParent<"ActionOp">,
                                 Terminator]> {
}

#endif // STANDALONE_OPS
