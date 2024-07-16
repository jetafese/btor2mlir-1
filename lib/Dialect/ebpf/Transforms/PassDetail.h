//===- PassDetail.h - GPU Pass class details --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT_TRANSFORMS_PASSDETAIL_H
#define EBPF_DIALECT_TRANSFORMS_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

namespace ebpf {

#define GEN_PASS_CLASSES
#include "Dialect/ebpf/Transforms/Passes.h.inc"

} // namespace ebpf
} // namespace mlir

#endif // EBPF_DIALECT_TRANSFORMS_PASSDETAIL_H
