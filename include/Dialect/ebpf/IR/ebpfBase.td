//===- ebpfDialect.td - ebpf dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT
#define EBPF_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ebpf constants definition.
//===----------------------------------------------------------------------===//

def ebpf_EQ  : I64EnumAttrCase<"eq", 0>;
def ebpf_NE  : I64EnumAttrCase<"ne", 1>;
def ebpf_SLT : I64EnumAttrCase<"slt", 2>;
def ebpf_SLE : I64EnumAttrCase<"sle", 3>;
def ebpf_SGT : I64EnumAttrCase<"sgt", 4>;
def ebpf_SGE : I64EnumAttrCase<"sge", 5>;
def ebpf_ULT : I64EnumAttrCase<"ult", 6>;
def ebpf_ULE : I64EnumAttrCase<"ule", 7>;
def ebpf_UGT : I64EnumAttrCase<"ugt", 8>;
def ebpf_UGE : I64EnumAttrCase<"uge", 9>;
def ebpf_SET : I64EnumAttrCase<"set", 10>;

def ebpfPredicateAttr : I64EnumAttr<
    "ebpfPredicate", "ebpf.cmp comparison predicate",
    [ebpf_EQ, ebpf_NE, ebpf_SLT, ebpf_SLE, ebpf_SGT, ebpf_SGE,
    ebpf_ULT, ebpf_ULE, ebpf_UGT, ebpf_UGE, ebpf_SET]> {
  let cppNamespace = "::mlir::ebpf";
}

//===----------------------------------------------------------------------===//
// ebpf dialect definition.
//===----------------------------------------------------------------------===//

def ebpf_Dialect : Dialect {
    let name = "ebpf";
    let summary = "An EBPF MLIR dialect";
    let cppNamespace = "::mlir::ebpf";
}

#endif // EBPF_DIALECT
