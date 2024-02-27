//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef BTOR_DIALECT_TRANSFORMS_PASSES_H
#define BTOR_DIALECT_TRANSFORMS_PASSES_H

#include "Dialect/Btor/Transforms/BtorLiveness.h"
#include "Dialect/Btor/Transforms/ResolveCasts.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace btor {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Btor/Transforms/Passes.h.inc"

} // namespace btor
} // namespace mlir

#endif // BTOR_DIALECT_TRANSFORMS_PASSES_H
