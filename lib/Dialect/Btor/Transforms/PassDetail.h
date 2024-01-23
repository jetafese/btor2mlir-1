//===- PassDetail.h - GPU Pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_DIALECT_TRANSFORMS_PASSDETAIL_H
#define BTOR_DIALECT_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace btor {

#define GEN_PASS_CLASSES
#include "Dialect/Btor/Transforms/Passes.h.inc"

} // namespace btor
} // namespace mlir

#endif // BTOR_DIALECT_TRANSFORMS_PASSDETAIL_H
