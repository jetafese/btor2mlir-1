//===- ebpfAttributes.h - ebpf dialect ops -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_IR_EBPF_ATTRS_H
#define EBPF_IR_EBPF_ATTRS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


//===----------------------------------------------------------------------===//
// ebpf Dialect Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "Dialect/ebpf/IR/ebpfAttributes.h.inc"

using namespace mlir;
using namespace mlir::ebpf;

#endif // EBPF_IR_EBPF_ATTRS_H