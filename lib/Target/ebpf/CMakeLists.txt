add_mlir_dialect_library(MLIRebpfTranslate
    ebpfToebpfIRTranslation.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Target/ebpf

	LINK_LIBS PUBLIC
	MLIRIR
	)
