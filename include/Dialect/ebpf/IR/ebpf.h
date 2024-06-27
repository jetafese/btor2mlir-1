//===- ebpfOps.h - ebpf dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_EBPFOPS_H
#define EBPF_EBPFOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ebpfDialect
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpfOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// ebpf Dialect Enum Attributes
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpfOpsEnums.h.inc"

//===----------------------------------------------------------------------===//
// ebpf Dialect Types
//===----------------------------------------------------------------------===//

#include "Dialect/ebpf/IR/ebpfTypes.h"

//===----------------------------------------------------------------------===//
// ebpf Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/ebpf/IR/ebpfOps.h.inc"

#endif // EBPF_EBPFOPS_H
