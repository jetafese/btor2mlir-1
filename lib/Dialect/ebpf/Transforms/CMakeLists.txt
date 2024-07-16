add_mlir_dialect_library(MLIRebpfTransforms
  ResolveMemory.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/ebpf/Transforms

  DEPENDS
  ebpfTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
