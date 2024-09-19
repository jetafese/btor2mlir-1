#!/bin/bash

### Update this path to the root of your prevail repository.
PREVAIL_ROOT=/Users/jetafese/Documents/code/btor2mlir/prevail
BTOR2MLIR_ROOT=/Users/jetafese/Documents/code/btor2mlir
SEAHORN_ROOT=/Users/jetafese/Documents/code/seahorn


EBPF_BENCHMARKS=${PREVAIL_ROOT}/ebpf-samples
PREVAIL_CHECK=${PREVAIL_ROOT}/check

BTOR2MLIR_DEBUG=${BTOR2MLIR_ROOT}/debug
BTOR2MLIR_TRANSLATE=${BTOR2MLIR_DEBUG}/bin/ebpf2mlir-translate
BTOR2MLIR_CHECK=${BTOR2MLIR_ROOT}/utils/ebpf/script.sh

DATE=$(date +"%m%d%y%H%M")
PREFIX=prevail_${DATE}
LOG=log_${DATE}

rm -f ${LOG}.txt
echo -n "Running ebpf pipeline on 369 programs ... "
echo "File,Result,trivial,seahorn_time,total"  1>> ${PREFIX}.csv
for f in ${EBPF_BENCHMARKS}/*/*.o
    do
    sections=($(${PREVAIL_CHECK} $f -l 2> /dev/null))
    for s in "${sections[@]}"
    do
        CLEANSECTION=${s//[^a-zA-Z0-9]/};
        rm -f $f.$CLEANSECTION.log.txt
        echo "${BTOR2MLIR_CHECK} ${f} ${BTOR2MLIR_DEBUG} ${SEAHORN_ROOT} ${s}" >> ${LOG}.txt
        echo -n $f:$s 1>> ${PREFIX}.csv
        /usr/bin/time ${BTOR2MLIR_CHECK} ${f} ${BTOR2MLIR_DEBUG} ${SEAHORN_ROOT} ${s} >> $f.$CLEANSECTION.time.txt 2>&1
        echo -n ",$(cat $f.$CLEANSECTION.log.txt)" >> ${PREFIX}.csv
        grep "maxresident" $f.$CLEANSECTION.time.txt >> $f.$CLEANSECTION.grep.txt
        echo -n ",$(cat $f.$CLEANSECTION.grep.txt)" >> ${PREFIX}.csv
        echo 1>> ${PREFIX}.csv
        rm -f $f.$CLEANSECTION.grep.txt
        rm -f $f.$CLEANSECTION.time.txt
        rm -f $f.$CLEANSECTION.log.txt
    done
done
echo "DONE"
