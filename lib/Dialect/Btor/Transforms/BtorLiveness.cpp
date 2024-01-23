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

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Analysis/Liveness.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace btor;

namespace {
LogicalResult resultIsLiveAfter(btor::WriteOp &op) {
    auto opPtr = op.getOperation();
    auto blockPtr = opPtr->getBlock();
    Value resValue = op.result();

    Liveness liveness(opPtr);

    // auto &allInValues = liveness.getLiveIn(&block);
    // auto &allOutValues = liveness.getLiveOut(blockPtr);
    // for (auto val : liveness.getLiveOut(blockPtr)) {
    //     val.dump();
    // }
    // auto allOperationsInWhichValueIsLive = liveness.resolveLiveness(resValue);
    bool isDeadAfter = liveness.isDeadAfter(resValue, opPtr);

    return isDeadAfter ? success() : failure();
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
        op.dump();
        LogicalResult status = llvm::TypeSwitch<Operation *, LogicalResult>(&op)
            // btor ops.
            .Case<btor::WriteOp>(
                [&](auto op) { return resultIsLiveAfter(op); })
            .Default([&](Operation *) {
                return success();
            });
        assert(status.succeeded());
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::btor::computeBtorLiveness() {
  return std::make_unique<BtorLivenessPass>();
}
