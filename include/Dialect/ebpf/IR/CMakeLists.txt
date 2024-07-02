set(LLVM_TARGET_DEFINITIONS ebpfOps.td)
add_mlir_dialect(ebpfOps ebpf)

set(LLVM_TARGET_DEFINITIONS ebpfAttributes.td)
mlir_tablegen(ebpfAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ebpfAttributes.cpp.inc -gen-attrdef-defs)
mlir_tablegen(ebpfOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ebpfOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(ebpfAttributesIncGen)

add_mlir_doc(ebpfAttributes ebpfAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(ebpfOps ebpfOps Dialects/ -gen-op-doc)
