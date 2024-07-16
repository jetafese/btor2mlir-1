//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT_TRANSFORMS_PASSES_H
#define EBPF_DIALECT_TRANSFORMS_PASSES_H

#include "Dialect/ebpf/Transforms/ResolveMemory.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ebpf {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/ebpf/Transforms/Passes.h.inc"

} // namespace ebpf
} // namespace mlir

#endif // EBPF_DIALECT_TRANSFORMS_PASSES_H
