// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

void foo(int x) {
  #pragma oss task reduction(+: x)
  {}
}

// CHECK: %0 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %x.addr), "QUAL.OSS.DEP.REDUCTION"(i32* %x.addr, i32* %x.addr, i64 4, i64 0, i64 4), "QUAL.OSS.DEP.REDUCTION.INIT"(i32* %x.addr, void (i32*, i32*, i64)* @red_init), "QUAL.OSS.DEP.REDUCTION.COMBINE"(i32* %x.addr, void (i32*, i32*, i64)* @red_comb) ]

// CHECK: define internal void @red_init(i32* %0, i32* %1, i64 %2)
// CHECK: store i32 0, i32* %3, align 4

// CHECK: define internal void @red_comb(i32* %0, i32* %1, i64 %2)
// CHECK: store i32 %add, i32* %3, align 4
