#ifndef P4C_P4DIALECTTOLLVM_H
#define P4C_P4DIALECTTOLLVM_H

#include <memory>
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::Pass> createP4DialectToLLVMPass();

#endif  // P4C_P4DIALECTTOLLVM_H
