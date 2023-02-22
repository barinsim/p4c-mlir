#ifndef BACKENDS_MLIR_TESTS_COMMON_H_
#define BACKENDS_MLIR_TESTS_COMMON_H_


#include <string>
#include "../cfgBuilder.h"


namespace p4mlir::tests {


BasicBlock* getByName(const std::map<const IR::IDeclaration*, BasicBlock*>&, const std::string&);
BasicBlock* getByStmtString(BasicBlock*, const std::string&);


} // p4mlir::tests

#endif /* BACKENDS_MLIR_TESTS_COMMON_H_ */
