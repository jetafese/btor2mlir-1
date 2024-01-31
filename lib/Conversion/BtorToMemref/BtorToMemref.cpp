#include "Conversion/BtorToMemref/ConvertBtorToMemrefPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::btor;

namespace {
LogicalResult shouldConvertToVector(Type &arrayType) {
  if (!arrayType.isa<MemRefType>()) {
    assert(arrayType.isa<VectorType>());
    return success(); /// the Vector pass will deal with this operation
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct InitArrayLowering
    : public ConvertOpToLLVMPattern<mlir::btor::InitArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::InitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = typeConverter->convertType(initArrayOp.getType());
    if (shouldConvertToVector(arrayType).succeeded()) {
      return success();
    }
    /// TODO
    // Operation *init = adaptor.init().getDefiningOp();
    // adaptor.init().dump();
    // assert(init->getName().getStringRef().equals(
    //     btor::ConstantOp::getOperationName()));
    // rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
    //     initArrayOp, arrayType, init->getAttr("value"));
    // rewriter.create<memref::GlobalOp>(initArrayOp.getLoc(), arrayType,
    //                                               adaptor.init());
    return success();
  }
};

struct ReadOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ReadOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ReadOp readOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = typeConverter->convertType(readOp.result().getType());
    auto arrayType = typeConverter->convertType(readOp.base().getType());
    if (shouldConvertToVector(arrayType).succeeded()) {
      return success();
    }
    /// TODO
    // rewriter.replaceOpWithNewOp<memref::LoadOp>(readOp, resType,
    // readOp.base(), readOp.index());
    return success();
  }
};

struct WriteInPlaceOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::WriteInPlaceOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::WriteInPlaceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteInPlaceOp writeInPlaceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType =
        typeConverter->convertType(writeInPlaceOp.result().getType());
    if (shouldConvertToVector(resType).succeeded()) {
      return success();
    }
    /// TODO
    // rewriter.replaceOpWithNewOp<memref::StoreOp>(
    //     writeInPlaceOp, resType, adaptor.value(), adaptor.base(),
    //     adaptor.index());
    return success();
  }
};

struct WriteOpLowering : public ConvertOpToLLVMPattern<mlir::btor::WriteOp> {
  using ConvertOpToLLVMPattern<mlir::btor::WriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::WriteOp writeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /// we are working under the assumption that all patterns are
    /// identified by the liveness analysis
    assert(writeOp.use_empty() && "include the liveness analysis pass");
    writeOp.erase();
    return success();
  }
};

struct IteWriteInPlaceOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::IteWriteInPlaceOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::IteWriteInPlaceOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::IteWriteInPlaceOp iteWriteInPlaceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType =
        typeConverter->convertType(iteWriteInPlaceOp.result().getType());
    if (shouldConvertToVector(resType).succeeded()) {
      return success();
    }
    /// TODO
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToMemrefConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLowering, WriteOpLowering, InitArrayLowering,
               WriteInPlaceOpLowering, IteWriteInPlaceOpLowering>(converter);
}

namespace {
struct ConvertBtorToMemrefPass
    : public ConvertBtorToMemrefBase<ConvertBtorToMemrefPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToMemrefConversionPatterns(converter, patterns);
    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::InitArrayOp>();

    // /// indexed operators
    target.addIllegalOp<btor::ReadOp>();
    target.addIllegalOp<btor::WriteOp>();
    target.addIllegalOp<btor::WriteInPlaceOp, btor::IteWriteInPlaceOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

/// Create a pass for lowering operations Btor operations
// to the Memref dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToMemrefPass() {
  return std::make_unique<ConvertBtorToMemrefPass>();
}