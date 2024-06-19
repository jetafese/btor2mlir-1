module {
  func @xdp_entry(%arg0: ui64, %arg1: ui64) -> ui64 {
    br ^bb1(%arg1: ui64)
  ^bb1(%1: ui64): 
    br ^bb2(%1: ui64)
  ^bb2(%2: ui64): 
    br ^bb3(%2: ui64)
  ^bb3(%3: ui64): 
    return %3 : ui64
  }
}
