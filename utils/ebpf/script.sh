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

# remove non-alphanumeric text from section name
CLEANSECTION=${SECTION//[^a-zA-Z0-9]/}

echo "$BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION $EBPF > $EBPF.$CLEANSECTION.mlir";
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION $EBPF > $EBPF.$CLEANSECTION.mlir; then 
    echo "error: translation failed" >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir;
    exit 1;
fi

echo "$BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.$CLEANSECTION.mlir > $EBPF.$CLEANSECTION.mlir.res.mlir";
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.$CLEANSECTION.mlir > $EBPF.$CLEANSECTION.mlir.res.mlir; then
    echo "error: analysis failed" >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir;
    rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir;

echo "$BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.$CLEANSECTION.mlir.res.mlir > $EBPF.$CLEANSECTION.mlir.opt";
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.$CLEANSECTION.mlir.res.mlir > $EBPF.$CLEANSECTION.mlir.opt; then
    echo "error: llvm dialect conversion failed" >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;
    rm -f $EBPF.$CLEANSECTION.mlir.opt;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;

echo "$BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.$CLEANSECTION.mlir.opt > $EBPF.$CLEANSECTION.mlir.opt.ll";
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.$CLEANSECTION.mlir.opt > $EBPF.$CLEANSECTION.mlir.opt.ll; then
    echo "error: llvmir conversion failed" >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;
    exit 1
fi
rm -f $EBPF.$CLEANSECTION.mlir.opt;

echo "$SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.$CLEANSECTION.mlir.opt.ll";
if ! timeout 15s $SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.$CLEANSECTION.mlir.opt.ll 1>> $EBPF.$CLEANSECTION.sea.txt; then
    echo "error: seahorn failed"  >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;
    rm -f $EBPF.$CLEANSECTION.sea.txt;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;

if ! grep "Result TRUE" $EBPF.$CLEANSECTION.sea.txt; then
    # execution reaching here means that the program is unsafe
    echo "0" >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.sea.txt;
    exit 1
fi

# execution returning safely means that the program is safe
rm -f $EBPF.$CLEANSECTION.sea.txt;