add_mlir_dialect_library(MLIRBtorTransforms
  BtorLiveness.cpp
  ResolveCasts.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Btor/Transforms

  DEPENDS
  BtorTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  )
