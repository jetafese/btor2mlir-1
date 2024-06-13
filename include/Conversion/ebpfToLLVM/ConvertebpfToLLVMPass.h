#ifndef EBPF_CONVERSION_EBPFTOLLVM_CONVERTEBPFTOLLVMPASS_H_
#define EBPF_CONVERSION_EBPFTOLLVM_CONVERTEBPFTOLLVMPASS_H_

#include <iostream>
#include <memory>
#include <utility>

#include "Dialect/ebpf/IR/ebpf.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
class ebpfToLLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace ebpf {

/// Collect a set of patterns to lower from ebpf to LLVM dialect
void populateebpfToLLVMConversionPatterns(ebpfToLLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

/// Creates a pass to convert the ebpf dialect into the LLVM dialect.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace ebpf
class ebpfToLLVMTypeConverter : public LLVMTypeConverter {
public:
  ebpfToLLVMTypeConverter(MLIRContext *ctx, bool to_LLVM = false,
                          const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, analysis) {}
};

} // namespace mlir

#endif // EBPF_CONVERSION_EBPFTOLLVM_CONVERTEBPFTOLLVMPASS_H_
