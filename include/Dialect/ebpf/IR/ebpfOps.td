//===- ebpfOps.td - ebpf dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EBPF_OPS
#define EBPF_OPS

include "ebpfBase.td"
// include "Dialect/Btor/IR/BtorBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
// include "Dialect/Btor/IR/BtorTypes.td"

//===----------------------------------------------------------------------===//
// Base ebpf operation definition.
//===----------------------------------------------------------------------===//

class ebpf_Op<string mnemonic, list<Trait> traits = []> :
        Op<ebpf_Dialect, mnemonic, traits>;

def CmpOp : ebpf_Op<"cmp", [
        NoSideEffect, SameTypeOperands,  
        TypesMatchWith<"result type has !bv<1> element type and same shape as operands", "lhs", 
        "result", "getI1SameShape($_self)">
        ] #ElementwiseMappable.traits> {
        let summary = "integer comparison operation";

        let description = [{
        The `cmp` operation is a generic comparison for two arguments 
        that need to have their types matching.

        Its first argument is an attribute that defines which type of comparison is
        performed. The following comparisons are supported:

        -   equal (mnemonic: `"eq"`; integer value: `0`)
        -   not equal (mnemonic: `"ne"`; integer value: `1`)
        -   signed less than (mnemonic: `"slt"`; integer value: `2`)
        -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
        -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
        -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
        -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
        -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
        -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
        -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

        The result is `1` if the comparison is true and `0` otherwise.

        Example:

        ```mlir
        // Custom form of scalar "signed less than" comparison.
        %x = ebpf.cmp "slt", %lhs, %rhs : !bv<1>
        ```
        }];

        let arguments = (ins ebpfPredicateAttr: $predicate,
                        SignlessIntegerLike: $lhs,
                        SignlessIntegerLike: $rhs);
        
        let results = (outs SignlessIntegerLike : $result);

        let builders = [OpBuilder<(ins "ebpfPredicate": $predicate, 
                                "Value": $lhs,
                                "Value": $rhs),
                        [{
                          build($_builder, $_state, ::getI1SameShape(lhs.getType()),
                                predicate, lhs, rhs);
                        }]>];
        
        let extraClassDeclaration = [{
                static StringRef getPredicateAttrName() { return "predicate"; }
                static ebpfPredicate getPredicateByName(StringRef name);

                ebpfPredicate getPredicate() {
                return (ebpfPredicate)(*this)->getAttrOfType<IntegerAttr>(
                        getPredicateAttrName()).getInt();
                }
        }];

        let verifier = [{ return verifyCmpOp(*this); }];
        let assemblyFormat = 
                "$predicate `,` $lhs `,` $rhs attr-dict `:` qualified(type($lhs))";
}

#endif // EBPF_OPS
