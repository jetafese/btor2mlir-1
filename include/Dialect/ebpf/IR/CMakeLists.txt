set(LLVM_TARGET_DEFINITIONS ebpfOps.td)
mlir_tablegen(ebpfOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ebpfOpsEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect(ebpfOps ebpf)

add_mlir_doc(ebpfOps ebpfOps Dialects/ -gen-op-doc)
