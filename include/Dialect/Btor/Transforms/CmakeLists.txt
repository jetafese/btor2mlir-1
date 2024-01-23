set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name BtorTransform)
add_public_tablegen_target(BtorTransformsIncGen)

add_mlir_doc(Passes BtorTransformPasses ./ -gen-pass-doc)
