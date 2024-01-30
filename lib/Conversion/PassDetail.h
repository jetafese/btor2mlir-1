#ifndef BTORCONVERSION_PASSDETAIL_H_
#define BTORCONVERSION_PASSDETAIL_H_

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace arith {
class ArithmeticDialect;
} // namespace arith

namespace btor {
class BtorDialect;
} // end namespace btor

namespace LLVM {
class LLVMDialect;
} // end namespace LLVM

namespace vector {
class VectorDialect;
} // end namespace vector

namespace memref {
class MemRefDialect;
}

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

} // end namespace mlir

#endif // BTORCONVERSION_PASSDETAIL_H_