set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ebpfTransform)
add_public_tablegen_target(ebpfTransformsIncGen)

add_mlir_doc(Passes ebpfTransformPasses ./ -gen-pass-doc)
