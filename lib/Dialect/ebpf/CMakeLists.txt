add_mlir_dialect_library(MLIRebpf
        IR/ebpfDialect.cpp
        IR/ebpfOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Dialect/ebpf

        DEPENDS
        MLIRebpfOpsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
