#ifndef STANDALONE_STANDALONEOPS_H
#define STANDALONE_STANDALONEOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#define GET_OP_CLASSES
#include "P4OpsEnumAttr.h.inc"
#include "P4OpsAttr.h.inc"
#include "P4Ops.h.inc"

#endif // STANDALONE_STANDALONEOPS_H
