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

using namespace mlir;
using namespace btor;

namespace {
struct BtorLivenessPass : public BtorLivenessBase<BtorLivenessPass> {
  void runOnOperation() override {
    Operation *rootOp = getOperation();
    auto module_regions = rootOp->getRegions();
    auto &blocks = module_regions.front().getBlocks();
    auto &funcOp = blocks.front().getOperations().front();
    // translate each block
    auto &regions = funcOp.getRegion(0);
    assert(regions.getBlocks().size() == 2);
    auto &nextBlock = regions.getBlocks().back();

    for (Operation &op : nextBlock.getOperations()) {
        if (op.getName().getStringRef() == btor::WriteOp::getOperationName()) {
            // Value arrayVal = op.getResult(0);
            Liveness liveness(&op);
            auto allOperationsInWhichValueIsLive = liveness.resolveLiveness(op.getResult(0));
            // bool isDeadAfter = liveness.isDeadAfter(adaptor.base(), writeOp);
            // assert (isDeadAfter);
            continue;
        }
        op.dump();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::btor::computeBtorLiveness() {
  return std::make_unique<BtorLivenessPass>();
}
