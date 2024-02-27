//===- BtorLiveness.h - Patterns for computing btor liveness ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_DIALECT_TRANSFORMS_RESOLVECASTS_H
#define BTOR_DIALECT_TRANSFORMS_RESOLVECASTS_H

#include "Dialect/Btor/IR/Btor.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
class Pass;

namespace btor {

/// Creates an instance of resolveCasts pass.
std::unique_ptr<mlir::Pass> resolveCasts();

} // namespace btor
} // namespace mlir

#endif // BTOR_DIALECT_TRANSFORMS_RESOLVECASTS_H