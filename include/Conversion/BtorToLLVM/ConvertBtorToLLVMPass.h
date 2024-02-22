#ifndef BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_
#define BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_

#include <memory>
#include <utility>
#include <iostream>

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
class BtorToLLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace btor {

/// Collect a set of patterns to lower from btor to LLVM dialect
void populateBtorToLLVMConversionPatterns(BtorToLLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);

/// Creates a pass to convert the Btor dialect into the LLVM dialect.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

} // namespace btor
class BtorToLLVMTypeConverter : public LLVMTypeConverter {
private:
  bool m_to_LLVM;
public:
  BtorToLLVMTypeConverter(MLIRContext *ctx, bool to_LLVM = false,
                          const DataLayoutAnalysis *analysis = nullptr)
      : LLVMTypeConverter(ctx, analysis), m_to_LLVM(to_LLVM) {
    addConversion([&](btor::BitVecType type) -> llvm::Optional<Type> {
      return convertBtorBitVecType(type);
    });
    addConversion([&](btor::ArrayType type) -> llvm::Optional<Type> {
      if (m_to_LLVM) {
        return convertBtorMemRefType(type);
      }
      return convertBtorArrayType(type);
    });
  }

  Type convertBtorBitVecType(btor::BitVecType type) {
    return ::IntegerType::get(type.getContext(), type.getWidth());
  }

  Type convertIntegerType(mlir::IntegerType type) {
    return btor::BitVecType::get(type.getContext(), type.getWidth());
  }

  Type convertBtorMemRefType(btor::ArrayType type) {
    auto memType = convertBtorArrayType(type);
    if (!memType.isa<MemRefType>()) {
      assert(memType.isa<VectorType>());
      return memType;
    }
    LLVMTypeConverter friendTypeConverter(type.getContext());
    auto result = friendTypeConverter.convertType(memType);
    assert(result.isa<LLVM::LLVMStructType>());
    std::cerr << "Converting array type to llvm struct: ";
      result.dump();
      std::cerr << std::endl;
    return result;
  }

  Type convertBtorArrayType(btor::ArrayType type) {
    std::cerr << "Converting array type to mem cell" << std::endl;
    unsigned indexWidth = pow(2, type.getShape().getWidth());
    auto elementType =
        ::IntegerType::get(type.getContext(), type.getElement().getWidth());
    if (type.getShape().getWidth() <= 5) {
      return VectorType::get(ArrayRef<int64_t>{indexWidth}, elementType);
    }
    return MemRefType::get(ArrayRef<int64_t>{indexWidth}, elementType);
  }
};

} // namespace mlir

#endif // BTOR_CONVERSION_BTORTOLLVM_CONVERTBTORTOLLVMPASS_H_
