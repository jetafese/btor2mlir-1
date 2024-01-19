#!/bin/bash

#1 - path to btor file
#2 - path to btor2mlir executables
#3 - path to SeaHorn executables
BTOR=$1
BTOR2MLIR=$2
SEAHORN=$3
# enable btorsim for validation
BTORTOOLS=$4

function usage () {
    echo "Usage: Give btor file, path to btor2mlir executables, path to SeaHorn executables"
    exit 1
}

if [ "$#" -ne 3 ]; then
    # if [ "$#" -ne 3 ]; then
    usage
    # fi
fi

echo "$BTOR2MLIR/btor2mlir-translate --import-btor $BTOR > $BTOR.mlir" ; \
$BTOR2MLIR/btor2mlir-translate --import-btor $BTOR > $BTOR.mlir
echo "$BTOR2MLIR/btor2mlir-translate --export-btor $BTOR.mlir > $BTOR.export.btor"
$BTOR2MLIR/btor2mlir-translate --export-btor $BTOR.mlir > $BTOR.export.btor ; \
echo "$BTOR2MLIR/btor2mlir-opt $BTOR.mlir \
        --convert-btornd-to-llvm \
        --convert-btor-to-vector \
        --convert-arith-to-llvm \
        --convert-std-to-llvm \
        --convert-btor-to-llvm \
        --convert-vector-to-llvm > $BTOR.mlir.opt" ; \
$BTOR2MLIR/btor2mlir-opt $BTOR.mlir \
        --convert-btornd-to-llvm \
        --convert-btor-to-vector \
        --convert-arith-to-llvm \
        --convert-std-to-llvm \
        --convert-btor-to-llvm \
        --convert-vector-to-llvm > $BTOR.mlir.opt ; \
echo "$BTOR2MLIR/btor2mlir-translate --mlir-to-llvmir $BTOR.mlir.opt > $BTOR.mlir.opt.ll"; \
$BTOR2MLIR/btor2mlir-translate --mlir-to-llvmir $BTOR.mlir.opt > $BTOR.mlir.opt.ll ; \

# exe-cex
# --oll=$BTOR.mlir.opt.ll.final.ll
echo "time timeout 300 $SEAHORN/sea yama -y configs/sea-cex.yaml bpf --verbose=2 -m64 $BTOR.mlir.opt.ll -o$BTOR.mlir.opt.ll.smt2"
time timeout 300 $SEAHORN/sea yama -y configs/sea-cex.yaml bpf --verbose=2 -m64 $BTOR.mlir.opt.ll -o$BTOR.mlir.opt.ll.smt2

echo "clang++-14 $BTOR.mlir.opt.ll /tmp/h2.ll ../../build/run/lib/libcex.a -o h2.out"
clang++-14 $BTOR.mlir.opt.ll /tmp/h2.ll ../../build/run/lib/libcex.a -o h2.out 

echo "./h2.out > /tmp/h2.txt"
env ./h2.out > /tmp/h2.txt

echo "python3 witness_generator.py /tmp/h2.txt"
python3 witness_generator.py /tmp/h2.txt

# echo "$BTORTOOLS/btorsim -v $BTOR cex.txt"
# $BTORTOOLS/btorsim -v $BTOR cex.txt
