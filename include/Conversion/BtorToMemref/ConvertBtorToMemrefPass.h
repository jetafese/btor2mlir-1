#ifndef BTOR_CONVERSION_BTORTOMEMREF_CONVERTBTORTOMEMREFPASS_H_
#define BTOR_CONVERSION_BTORTOMEMREF_CONVERTBTORTOMEMREFPASS_H_

#include "Conversion/BtorToLLVM/ConvertBtorToLLVMPass.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <memory>

namespace mlir {
class Pass;

class RewritePatternSet;

namespace btor {
/// Collect a set of patterns to lower from btor to memref dialect
void populateBtorToMemrefConversionPatterns(BtorToLLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);

/// Creates a pass to convert the Btor dialect into the memref dialect.
std::unique_ptr<Pass> createLowerToMemrefPass();

} // namespace btor
} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOMEMREF_CONVERTBTORTOMEMREFPASS_H_
