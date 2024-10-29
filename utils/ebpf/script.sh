#!/bin/bash

# 1 - path to ebpf file
# 2 - path to ebpf2mlir build directory
# 3 - path to SeaHorn executables
# 4 - desired section in ebpf
EBPF=$1
BTOR2MLIR=$2
SEAHORN=$3
SECTION=$4
FUNCTION=$5

function usage () {
    echo "Usage: Give ebpf file," \
        "path to ebpf2mlir build directory," \
        "path to SeaHorn executables," \
        "desired section" \
        "desired function"
    exit 1
}

if [ "$#" -ne 5 ]; then
    usage
fi

# remove non-alphanumeric text from section name
CLEANSECTION=${SECTION//[^a-zA-Z0-9]/}

echo "$BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION --function $FUNCTION $EBPF > $EBPF.$CLEANSECTION.mlir";
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --import-ebpf-mem --section $SECTION --function $FUNCTION $EBPF > $EBPF.$CLEANSECTION.mlir; then 
    echo "error: translation failed,," >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir;
    exit 1;
fi

echo "$BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.$CLEANSECTION.mlir > $EBPF.$CLEANSECTION.mlir.res.mlir";
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --inline --resolve-mem  $EBPF.$CLEANSECTION.mlir > $EBPF.$CLEANSECTION.mlir.res.mlir; then
    echo "error: analysis failed,," >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir;
    rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir;

echo "$BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.$CLEANSECTION.mlir.res.mlir > $EBPF.$CLEANSECTION.mlir.opt";
if ! $BTOR2MLIR/bin/ebpf2mlir-opt --convert-ebpf-to-llvm --reconcile-unrealized-casts  $EBPF.$CLEANSECTION.mlir.res.mlir > $EBPF.$CLEANSECTION.mlir.opt; then
    echo "error: to llvm dialect conversion failed,," >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;
    rm -f $EBPF.$CLEANSECTION.mlir.opt;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir.res.mlir;

echo "$BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.$CLEANSECTION.mlir.opt > $EBPF.$CLEANSECTION.mlir.opt.ll";
if ! $BTOR2MLIR/bin/ebpf2mlir-translate --mlir-to-llvmir $EBPF.$CLEANSECTION.mlir.opt > $EBPF.$CLEANSECTION.mlir.opt.ll; then
    echo "error: final conversion failed,," >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;
    exit 1
fi
rm -f $EBPF.$CLEANSECTION.mlir.opt;

echo "/usr/bin/llvm-link-14 $EBPF.$CLEANSECTION.mlir.opt.ll $BTOR2MLIR/helper_summaries.ll -S -o $EBPF.$CLEANSECTION.mlir.opt.sum.ll";
if ! /usr/bin/llvm-link-14 $EBPF.$CLEANSECTION.mlir.opt.ll $BTOR2MLIR/helper_summaries.ll -S -o $EBPF.$CLEANSECTION.mlir.opt.sum.ll; then
    echo "error: linking failed,," >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt.sum.ll
    rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;
    exit 1
fi
rm -f $EBPF.$CLEANSECTION.mlir.opt.ll;

echo "$SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.$CLEANSECTION.mlir.opt.sum.ll";
if ! timeout 150s $SEAHORN/build/run/bin/sea yama -y $BTOR2MLIR/../utils/cex/witness/configs/sea-cex.yaml fpf $EBPF.$CLEANSECTION.mlir.opt.sum.ll >> $EBPF.$CLEANSECTION.sea.txt 2>&1; then
    echo "error: seahorn timeout,,"  >> $EBPF.$CLEANSECTION.log.txt;
    rm -f $EBPF.$CLEANSECTION.mlir.opt.sum.ll;
    rm -f $EBPF.$CLEANSECTION.sea.txt;
    exit 1;
fi
rm -f $EBPF.$CLEANSECTION.mlir.opt.sum.ll;

if grep "WARNING: no assertion was found" $EBPF.$CLEANSECTION.sea.txt; then
    # safe, trivial
    echo -n "1,1," >> $EBPF.$CLEANSECTION.log.txt;
    grep "seahorn_total" $EBPF.$CLEANSECTION.sea.txt >> $EBPF.$CLEANSECTION.log.txt
    rm -f $EBPF.$CLEANSECTION.sea.txt;
    exit 0
fi

if ! grep "Result TRUE" $EBPF.$CLEANSECTION.sea.txt; then
    # execution reaching here means that the program is unsafe
    echo -n "0,0," >> $EBPF.$CLEANSECTION.log.txt;
    grep "seahorn_total" $EBPF.$CLEANSECTION.sea.txt >> $EBPF.$CLEANSECTION.log.txt
    rm -f $EBPF.$CLEANSECTION.sea.txt;
    exit 1
fi

# execution returning safely means that the program is safe
echo -n "1,0," >> $EBPF.$CLEANSECTION.log.txt;
grep "seahorn_total" $EBPF.$CLEANSECTION.sea.txt >> $EBPF.$CLEANSECTION.log.txt
rm -f $EBPF.$CLEANSECTION.sea.txt;