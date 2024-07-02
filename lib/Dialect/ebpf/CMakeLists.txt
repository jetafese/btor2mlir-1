add_mlir_dialect_library(MLIRebpf
        IR/ebpfDialect.cpp
        IR/ebpfOps.cpp
        IR/ebpfTypes.cpp
        IR/ebpfAttributes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/ebpf

        DEPENDS
        MLIRebpfOpsIncGen
        ebpfAttributesIncGen

	LINK_LIBS PUBLIC
  MLIRDialect
	MLIRIR
	)
