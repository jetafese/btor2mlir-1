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
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"

//===----------------------------------------------------------------------===//
// Base ebpf operation definition.
//===----------------------------------------------------------------------===//

class ebpf_Op<string mnemonic, list<Trait> traits = []> :
        Op<ebpf_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ebpf binary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for ebpf arithmetic operations.  Requires operands and
// results to be of the same type, but does not constrain them to specific
// types.
class ebpfArithmeticOp<string mnemonic, list<Trait> traits = []>
    : Op<ebpf_Dialect, mnemonic, traits>;

// This operation takes two operands and returns one result,
// each of these is required to be of the same type.
//  The custom assembly form of the operation is as follows
//
//     <op> %0, %1 : i64
// clang-format off
class ebpfBinaryOp<string mnemonic, list<Trait> traits = []>
  : ebpfArithmeticOp<mnemonic,
                      !listconcat(traits, [SameOperandsAndResultType])>,
    Arguments<(ins SignlessIntegerLike: $lhs, 
                SignlessIntegerLike: $rhs)>,
    Results<(outs SignlessIntegerLike: $result)> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  // clang-format on
}

def AddOp : ebpfBinaryOp<"add", [Commutative]> {
  let summary = "integer addition operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.add %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def SubOp : ebpfBinaryOp<"sub"> {
  let summary = "integer subtraction operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.sub %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def MulOp : ebpfBinaryOp<"mul", [Commutative]> {
  let summary = "integer multiplication operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.mul %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def SDivOp : ebpfBinaryOp<"sdiv"> {
  let summary = "integer signed division operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.sdiv %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}


def UDivOp : ebpfBinaryOp<"udiv"> {
  let summary = "integer unsigned division operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.udiv %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def SModOp : ebpfBinaryOp<"smod"> {
  let summary = "integer signed modulus operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.smod %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def UModOp : ebpfBinaryOp<"umod"> {
  let summary = "integer unsigned modulus operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.umod %lhs, %rhs: i64
    ```
  }];
  // clang-format on
}

def OrOp : ebpfBinaryOp<"or", [Commutative]> {
  let summary = "integer binary and operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.or %lhs, %rhs: !bv<32>
    ```
  }];
  // clang-format on
}

def AndOp : ebpfBinaryOp<"and", [Commutative]> {
  let summary = "integer binary and operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.and %lhs, %rhs: !bv<32>
    ```
  }];
  // clang-format on
}

def LSHOp : ebpfBinaryOp<"lsh"> {
  let summary = "integer left logical shift binary operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.lsh %lhs, %rhs: !bv<32>
    ```
  }];
  // clang-format on
}

def RSHOp : ebpfBinaryOp<"rsh"> {
  let summary = "integer right logical shift operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
      ```mlir
      %res = ebpf.rsh %lhs, %rhs: !bv<32>
      ```
  }];
  // clang-format on
}

def ShiftRAOp : ebpfBinaryOp<"arsh"> {
  let summary = "integer right arithmetic shift operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.arsh %lhs, %rhs: !bv<32>
    ```
  }];
  // clang-format on
}

def XOrOp : ebpfBinaryOp<"xor", [Commutative]> {
  let summary = "integer binary xor operation";
  // clang-format off
  let description = [{
    This operation takes two integer arguments and returns an integer.

    Example:
        
    ```mlir
    %res = ebpf.xor %lhs, %rhs: !bv<32>
    ```
  }];
  // clang-format on
}

//===----------------------------------------------------------------------===//
// ebpf unary integer ops definitions
//===----------------------------------------------------------------------===//

// Base class for unary ops. Requires single operand and result. Individual
// classes will have `operand` accessor.
// clang-format off
class ebpfUnaryOp<string mnemonic, list<Trait> traits = []>
    : Op<ebpf_Dialect, mnemonic,
         !listconcat(traits, [NoSideEffect, SameOperandsAndResultType])>,
      Arguments<(ins SignlessIntegerLike: $operand)>,
      Results<(outs SignlessIntegerLike: $result)> {

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
  // clang-format on
}

def NegOp : ebpfUnaryOp<"neg"> {
  let summary = "integer negation";
  // clang-format off
  let description = [{
    The `neg` operation computes the negation of a given value.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    // Scalar negation value.
    %a = ebpf.neg %b: i64
    ```
  }];
  // clang-format on
}

def MoveOp : ebpfUnaryOp<"move"> {
  let summary = "integer 64bit assignment";
  // clang-format off
  let description = [{
    The `move` operation assigns a 64bit value to another register.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.move %b: i64
    ```
  }];
  // clang-format on
}

def Move32Op : ebpfUnaryOp<"move32"> {
  let summary = "integer 32bit assignment";
  // clang-format off
  let description = [{
    The `move32` operation assigns a 32bit value to another register.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.move %b: i64
    ```
  }];
  // clang-format on
}

def Move16Op : ebpfUnaryOp<"move16"> {
  let summary = "integer 16bit assignment";
  // clang-format off
  let description = [{
    The `move32` operation assigns a 16bit value to another register.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.move %b: i64
    ```
  }];
  // clang-format on
}

def Move8Op : ebpfUnaryOp<"move8"> {
  let summary = "integer 8bit assignment";
  // clang-format off
  let description = [{
    The `move8` operation assigns a 8bit value to another register.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.move %b: i64
    ```
  }];
  // clang-format on
}

def BE16 : ebpfUnaryOp<"be16"> {
  let summary = "integer big endian byte order";
  // clang-format off
  let description = [{
    The `htobe16` operation assigns host bits to big endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.be16 %b: i64
    ```
  }];
  // clang-format on
}

def BE32 : ebpfUnaryOp<"be32"> {
  let summary = "integer big endian byte order";
  // clang-format off
  let description = [{
    The `htobe32` operation assigns host bits to big endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.be32 %b: i64
    ```
  }];
  // clang-format on
}

def BE64 : ebpfUnaryOp<"be64"> {
  let summary = "integer big endian byte order";
  // clang-format off
  let description = [{
    The `htobe64` operation assigns host bits to big endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.be64 %b: i64
    ```
  }];
  // clang-format on
}

def LE16 : ebpfUnaryOp<"le16"> {
  let summary = "integer little endian byte order";
  // clang-format off
  let description = [{
    The `htobe16` operation assigns host bits to little endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.le16 %b: i16
    ```
  }];
  // clang-format on
}

def LE32 : ebpfUnaryOp<"le32"> {
  let summary = "integer little endian byte order";
  // clang-format off
  let description = [{
    The `htobe32` operation assigns host bits to little endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.le32 %b: i32
    ```
  }];
  // clang-format on
}

def LE64 : ebpfUnaryOp<"le64"> {
  let summary = "integer little endian byte order";
  // clang-format off
  let description = [{
    The `htobe64` operation assigns host bits to little endian.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.le64 %b: i64
    ```
  }];
  // clang-format on
}

def SWAP16 : ebpfUnaryOp<"swap16"> {
  let summary = "integer swap operation";
  // clang-format off
  let description = [{
    The `bwsap16` operation unconditionally reverses the 
    order of bytes.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.bswap16 %b: i64
    ```
  }];
  // clang-format on
}

def SWAP32 : ebpfUnaryOp<"swap32"> {
  let summary = "integer swap operation";
  // clang-format off
  let description = [{
    The `bwsap32` operation unconditionally reverses the 
    order of bytes.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.bswap32 %b: i64
    ```
  }];
  // clang-format on
}

def SWAP64 : ebpfUnaryOp<"swap64"> {
  let summary = "integer swap operation";
  // clang-format off
  let description = [{
    The `bwsap64` operation unconditionally reverses the 
    order of bytes.
    It takes one operand and returns one result of the same type.

    Example:

    ```mlir
    %a = ebpf.bswap64 %b: i64
    ```
  }];
  // clang-format on
}

#endif // EBPF_OPS
