//===- ebpfDialect.td - ebpf dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_DIALECT
#define EBPF_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ebpf dialect definition.
//===----------------------------------------------------------------------===//

def ebpf_Dialect : Dialect {
    let name = "ebpf";
    let summary = "An EBPF MLIR dialect";
    let cppNamespace = "::mlir::ebpf";
    let extraClassDeclaration = [{
    private:
            // Register the custom ebpf Types.
            void registerTypes();

            // Register the custom btor attributes.
            void registerAttrs();
    public:
    }];
}

#endif // EBPF_DIALECT
