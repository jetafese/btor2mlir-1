module {
  func @main() {
    // create context and store address in r1
    // setup stack and store address in r10
    call @xdp_entry(%r1, %r10)
  }
  func @xdp_entry(%r1_0: i64, %r10_0: i64) {
  // inst 0 - 5
    %r0_1 = 2
    %r2_1 = ldxw %r1 4
    %r1_1 = ldxw %r1 0
    %r3_1 = %r1_1
    %r3_2 = add %r3_1 14
    %cond_1 = cmp gt %r3_2 %r2_1
    cond_br %cond_1 ^bb10(%r0_1) ^bb1(%r0_1, %r1_1, %r2_1, %r3_2, %r10) 
  ^bb1(%r0, %r1, %r2, %r3, %r10):
  // inst 6 - 13
    %r4_1 = ldxb %r1 12
    %r3_3 = ldxb %r1 13
    %r3_4 = lsh  %r3_3 8
    %r3_5 = or  %r3_4 %r4_1
    %r4_2 = %r3_5
    %r4_3 = and %r4_2 255
    %r5_1 = 6
    %cond_2 = cmp gt %r5_1 %r4_3
    cond_br %cond_2 ^bb10(%r0) ^bb2(%r0, %r1, %r2, %r3_5, %r4_3, %r5_1, %r10)
  ^bb2(%r0, %r1, %r2, %r3, %r4, %r5, %r10);
  // inst 14
    %cond_3 = cmp eq %r3 43144
    cond_br %cond_3 ^bb4(%r0) ^bb3(%r0, %r1, %r2, %r3, %r4, %r5, %r10)
  ^bb3(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 15 - 16
    %r4_4 = 14
    %cond_4 = cmp neq %r3_5 129
    cond_br %cond_4 ^bb6(%r0) ^bb4(%r0, %r1, %r2, %r3, %r4_4, %r5, %r10)
  ^bb4(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 17 - 19
    %r3_6 = %r1
    %r3_7 = %r3 18
    %cond_5 = cmp gt %r3_7 %r2
    cond_br %cond_5 ^bb10(%r0) ^bb5(%r0, %r1, %r2, %r3_7, %r4, %r5, %r10)    
  ^bb5(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 20 - 22
    %r4_5 = 18
    %r3_8 = ldxh %r1 16
    br ^bb6(%r0, %r1, %r2, %r3_8, %r4_5, %r5, %r10)
  ^bb6(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 22 - 23
    %r3_9 = and %r3 65535
    %cond_6 = cmp neq %r3_9 8
    cond_br %cond_6 ^bb10(%r0) ^bb7(%r0, %r1, %r2, %r3_9, %r4, %r5, %r10)
  ^bb7(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 24 - 28
    %r1_2 = add %r1 %r4
    %r0_2 = 0
    %r3_10 = %r1_2
    %r3_11 = add %r3_10 20
    %cond_7 = cmp gt %r3_11 %r2
    cond_br %cond_7 ^bb10(%r0_2) ^bb8(%r0_2, %r1_2, %r2, %r3_11, %r4, %r5, %r10)
  ^bb7(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 29 - 38
    %r1_3 = ldxb %r1 8
    stw %r10_0 -4 %r1_3
    %r2_2 = %r10
    %r2_3 = add %r2_2 -4
    %r1_4 = readmap 1 %r2_3
    %r1_5 = %r0
    %r0_3 = 2
    %cond_8 = cmp eq %r1_5 0
    cond_br %cond_8 ^bb10(%r0_3) ^bb9(%r0_3, %r1_5, %r2_3, %r3, %r4, %r5, %r10) 
  ^bb9(%r0, %r1, %r2, %r3, %r4, %r5, %r10):
  // inst 39 -41
    %r2_4 = ldxdw %r1 0
    %r2_5 = add %r2 1
    stdw %r1 0 %r2_5
    br ^bb10(%r0)
  ^bb10(%r_0):
  // inst 42
    return %r0
  }
}