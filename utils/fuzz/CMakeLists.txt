add_library(fuzz
  nd_fuzz.c)

install (TARGETS fuzz
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
  