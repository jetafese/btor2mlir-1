#include "Conversion/BtorToVector/ConvertBtorToVectorPass.h"
#include "Dialect/Btor/IR/Btor.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::btor;

namespace {
LogicalResult shouldConvertToMemRef(Type &arrayType) {
  if (!arrayType.isa<VectorType>()) {
    assert(arrayType.isa<MemRefType>());
    return success(); /// the MemRef pass will deal with this operation
  }
  return failure();
}

unsigned numConcats(unsigned opWidth, unsigned ndFunc = 32) {
  if (opWidth <= ndFunc) {
    return 0;
  }
  if ((opWidth % ndFunc) == 0) {
    return opWidth / ndFunc;
  }
  return (opWidth / ndFunc) + 1;
}

template <typename Op>
Value getNDValueHelper(Op op, mlir::ConversionPatternRewriter &rewriter,
                       ModuleOp module, Type resultType, unsigned ndSize = 32) {
  const std::string havoc = "nd_bv32";
  auto loc = op.getLoc();
  auto havocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(havoc);
  if (!havocFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto havocFuncTy =
        LLVM::LLVMFunctionType::get(rewriter.getIntegerType(ndSize), {});
    havocFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                  havoc, havocFuncTy);
  }
  Value callND =
      rewriter.create<LLVM::CallOp>(loc, havocFunc, llvm::None).getResult(0);
  // Type resultWidthType = rewriter.getI32Type();
  // unsigned resultWidth = ndSize;
  // // concatenate as many 32bit integers as needed
  auto concats = numConcats(resultType.getIntOrFloatBitWidth());
  Value finalVal;
  if (concats == 0) {
    finalVal =
        rewriter.create<LLVM::TruncOp>(loc, TypeRange({resultType}), callND);
  } else {
    finalVal = rewriter.create<LLVM::ZExtOp>(loc, resultType, callND);
  }
  return finalVal;
}

//===----------------------------------------------------------------------===//
// Lowering Declarations
//===----------------------------------------------------------------------===//

struct ArrayOpLowering : public ConvertOpToLLVMPattern<mlir::btor::ArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::ArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::ArrayOp arrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = typeConverter->convertType(arrayOp.getType());
    if (shouldConvertToMemRef(arrayType).succeeded()) {
      return success();
    }
    auto vecType = arrayType.cast<VectorType>();
    auto module = arrayOp->getParentOfType<ModuleOp>();
    auto callND =
        getNDValueHelper(arrayOp, rewriter, module, vecType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::btor::VectorInitArrayOp>(
        arrayOp, arrayType, callND);
    return success();
  }
};

struct InitArrayLowering
    : public ConvertOpToLLVMPattern<mlir::btor::InitArrayOp> {
  using ConvertOpToLLVMPattern<mlir::btor::InitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::InitArrayOp initArrayOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arrayType = typeConverter->convertType(initArrayOp.getType());
    if (shouldConvertToMemRef(arrayType).succeeded()) {
      return success();
    }
    rewriter.replaceOpWithNewOp<mlir::btor::VectorInitArrayOp>(
        initArrayOp, arrayType, adaptor.init());
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
    if (shouldConvertToMemRef(arrayType).succeeded()) {
      return success();
    }
    rewriter.replaceOpWithNewOp<mlir::btor::VectorReadOp>(
        readOp, resType, adaptor.base(), adaptor.index());
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
    if (shouldConvertToMemRef(resType).succeeded()) {
      return success();
    }
    rewriter.replaceOpWithNewOp<mlir::btor::VectorWriteOp>(
        writeInPlaceOp, resType, adaptor.value(), adaptor.base(),
        adaptor.index());
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
    if (shouldConvertToMemRef(resType).succeeded()) {
      return success();
    }
    Value writeInPlace = rewriter.create<mlir::btor::VectorWriteOp>(
        iteWriteInPlaceOp.getLoc(), resType, adaptor.value(), adaptor.base(),
        adaptor.index());
    rewriter.replaceOpWithNewOp<LLVM::SelectOp>(iteWriteInPlaceOp, resType,
                                                adaptor.condition(),
                                                writeInPlace, adaptor.base());
    return success();
  }
};

struct VectorInitArrayOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorInitArrayOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorInitArrayOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorInitArrayOp vecInitArrayOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(
        vecInitArrayOp, vecInitArrayOp.getType(), vecInitArrayOp.init());
    return success();
  }
};

struct VectorReadOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorReadOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorReadOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorReadOp vecReadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(
        vecReadOp, vecReadOp.result().getType(), adaptor.base(),
        adaptor.index());
    return success();
  }
};

struct VectorWriteOpLowering
    : public ConvertOpToLLVMPattern<mlir::btor::VectorWriteOp> {
  using ConvertOpToLLVMPattern<
      mlir::btor::VectorWriteOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(mlir::btor::VectorWriteOp vecWriteOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<vector::InsertElementOp>(
        vecWriteOp, adaptor.base().getType(), adaptor.value(), adaptor.base(),
        adaptor.index());
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Populate Lowering Patterns
//===----------------------------------------------------------------------===//

void mlir::btor::populateBtorToVectorConversionPatterns(
    BtorToLLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ReadOpLowering, WriteOpLowering, InitArrayLowering,
               WriteInPlaceOpLowering, VectorInitArrayOpLowering,
               VectorReadOpLowering, VectorWriteOpLowering,
               IteWriteInPlaceOpLowering, ArrayOpLowering>(converter);
}

namespace {
struct ConvertBtorToVectorPass
    : public ConvertBtorToVectorBase<ConvertBtorToVectorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMConversionTarget target(getContext());
    BtorToLLVMTypeConverter converter(&getContext());

    mlir::btor::populateBtorToVectorConversionPatterns(converter, patterns);
    mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    /// Configure conversion to lower out btor; Anything else is fine.
    // init operators
    target.addIllegalOp<btor::InitArrayOp, btor::VectorInitArrayOp>();
    target.addIllegalOp<btor::ArrayOp>();

    /// indexed operators
    target.addIllegalOp<btor::ReadOp, btor::VectorReadOp>();
    target.addIllegalOp<btor::WriteOp, btor::VectorWriteOp>();
    target.addIllegalOp<btor::WriteInPlaceOp, btor::IteWriteInPlaceOp>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

/// Create a pass for lowering operations the remaining `Btor` operations
// to the Vector dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::btor::createLowerToVectorPass() {
  return std::make_unique<ConvertBtorToVectorPass>();
}
