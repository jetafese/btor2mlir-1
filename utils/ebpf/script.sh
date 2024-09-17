#!/bin/bash

# 1 - path to ebpf file
# 2 - path to ebpf2mlir build directory
# 3 - path to SeaHorn executables
# 4 - desired section in ebpf
EBPF=$1
BTOR2MLIR=$2
SEAHORN=$3
SECTION=$4

function usage () {
    echo "Usage: Give ebpf file," \
        "path to ebpf2mlir build directory," \
        "path to SeaHorn executables," \
        "desired section"
    exit 1
}

if [ "$#" -ne 4 ]; then
    usage
fi

echo "$BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION $EBPF > $EBPF.mlir" >> $EBPF.log.txt;
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION $EBPF > $EBPF.mlir; then 
    echo "error: translation failed" >> $EBPF.log.txt
    exit 1
fi
echo "$BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.mlir > $EBPF.mlir.res.mlir"  >> $EBPF.log.txt;
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.mlir > $EBPF.mlir.res.mlir; then
    echo "error: analysis failed" >> $EBPF.log.txt
    exit 1
fi
echo "$BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.mlir.res.mlir > $EBPF.mlir.opt"  >> $EBPF.log.txt;
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.mlir.res.mlir > $EBPF.mlir.opt; then
    echo "error: llvm dialect conversion failed" >> $EBPF.log.txt
    exit 1
fi
echo "$BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.mlir.opt > $EBPF.mlir.opt.ll"  >> $EBPF.log.txt;
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.mlir.opt > $EBPF.mlir.opt.ll; then
    echo "error: llvmir conversion failed" >> $EBPF.log.txt
    exit 1
fi
echo "$SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.mlir.opt.ll -o$EBPF.smt2" >> $EBPF.log.txt;
if ! $SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.mlir.opt.ll -o$EBPF.smt2 1>> $EBPF.log.txt; then
    echo "error: seahorn failed"  >> $EBPF.log.txt
    exit 1
fi

if ! grep "Result TRUE" $EBPF.log.txt; then
    echo "error: program is unsafe"
    exit 1
fi
