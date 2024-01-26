//===- BtorLiveness.cpp - Standard Function conversions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Btor/Transforms/BtorLiveness.h"
#include "Dialect/Btor/IR/Btor.h"
#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace btor;

namespace {
LogicalResult replaceWithWriteInPlace(btor::WriteOp &op) {
  auto status = resultIsLiveAfter(op);
  if (status.succeeded()) {
    auto resValue = op.result();
    auto opPtr = op.getOperation();
    auto m_context = opPtr->getContext();
    auto m_builder = OpBuilder(m_context);
    m_builder.setInsertionPointAfterValue(resValue);
    Value writeInPlace = m_builder.create<btor::WriteInPlaceOp>(
        op.getLoc(), op.getType(), op.value(), op.base(), op.index());
    resValue.replaceAllUsesWith(writeInPlace);
    assert(resValue.use_empty());
  }

  return status;
}

struct BtorLivenessPass : public BtorLivenessBase<BtorLivenessPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    auto module_regions = rootOp->getRegions();
    auto &blocks = module_regions.front().getBlocks();
    auto &funcOp = blocks.front().getOperations().front();
    auto &regions = funcOp.getRegion(0);
    assert(regions.getBlocks().size() == 2);
    auto &nextBlock = regions.getBlocks().back();

    for (Operation &op : nextBlock.getOperations()) {
      LogicalResult status =
          llvm::TypeSwitch<Operation *, LogicalResult>(&op)
              // btor ops.
              .Case<btor::WriteOp>(
                  [&](auto op) { return replaceWithWriteInPlace(op); })
              .Default([&](Operation *) { return success(); });
      assert(status.succeeded());
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::btor::computeBtorLiveness() {
  return std::make_unique<BtorLivenessPass>();
}

/// @brief Determine if a writeOp is used by non-branch operations
/// @param Btor WriteOp
/// @return success/failure wrapped in LogicalResult
LogicalResult mlir::btor::resultIsLiveAfter(btor::WriteOp &op) {
  auto opPtr = op.getOperation();
  auto blockPtr = opPtr->getBlock();
  Value resValue = op.result();

  assert(opPtr != nullptr);
  assert(blockPtr != nullptr);
  assert(!resValue.isUsedOutsideOfBlock(blockPtr));
  if (!resValue.hasOneUse()) {
    return failure();
  }
  auto use = resValue.user_begin();
  auto useOp = use.getCurrent().getUser();
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(useOp)
          .Case<mlir::BranchOp>([&](auto op) { return success(); })
          .Default([&](Operation *) { return failure(); });
  return status;
}
