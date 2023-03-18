#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "P4Dialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def ConstantOp : P4Op<"constant"> {
    let arguments = (ins  I32Attr:$value);
    let results = (outs I32:$result);
    let hasVerifier = 1;

    let builders = [
    // Build a constant with a given constant int value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "int":$value)>
  ];
}

#endif // STANDALONE_OPS
