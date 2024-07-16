//===- ResolveMemory.h - Patterns for resolving ebpf memory ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT_TRANSFORMS_RESOLVEMEMORY_H
#define EBPF_DIALECT_TRANSFORMS_RESOLVEMEMORY_H

#include "Dialect/ebpf/IR/ebpf.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
class Pass;

namespace ebpf {

/// Creates an instance of resolveCasts pass.
std::unique_ptr<mlir::Pass> resolveMemory();

} // namespace ebpf
} // namespace mlir

#endif // EBPF_DIALECT_TRANSFORMS_RESOLVEMEMORY_H