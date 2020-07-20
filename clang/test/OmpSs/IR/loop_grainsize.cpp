// RUN: %clang_cc1 -verify -fompss-2 -disable-llvm-passes -ferror-limit 100 %s -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics
template<typename T> T foo() { return T(); }

void bar(int n) {
    int vla[n];
    #pragma oss taskloop grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(foo<int>())
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(n)
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop grainsize(vla[1])
    for (int i = 0; i < 10; ++i) {}
    #pragma oss taskloop for grainsize(vla[1])
    for (int i = 0; i < 10; ++i) {}
}

// CHECK: %3 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i), "QUAL.OSS.LOOP.IND.VAR"(i32* %i), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %call), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %call) ]
// CHECK: %4 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i1), "QUAL.OSS.LOOP.IND.VAR"(i32* %i1), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %call2), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %call2) ]
// CHECK: %6 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i3), "QUAL.OSS.LOOP.IND.VAR"(i32* %i3), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %5), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %5) ]
// CHECK: %8 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i4), "QUAL.OSS.LOOP.IND.VAR"(i32* %i4), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %7), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %7) ]
// CHECK: %10 = call token @llvm.directive.region.entry() [ "DIR.OSS"([9 x i8] c"TASKLOOP\00"), "QUAL.OSS.PRIVATE"(i32* %i5), "QUAL.OSS.LOOP.IND.VAR"(i32* %i5), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %9), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %9) ]
// CHECK: %12 = call token @llvm.directive.region.entry() [ "DIR.OSS"([13 x i8] c"TASKLOOP.FOR\00"), "QUAL.OSS.PRIVATE"(i32* %i6), "QUAL.OSS.LOOP.IND.VAR"(i32* %i6), "QUAL.OSS.LOOP.LOWER.BOUND"(i32 0), "QUAL.OSS.LOOP.UPPER.BOUND"(i32 10), "QUAL.OSS.LOOP.STEP"(i32 1), "QUAL.OSS.LOOP.GRAINSIZE"(i32 %11), "QUAL.OSS.LOOP.TYPE"(i64 0), "QUAL.OSS.CAPTURED"(i32 0, i32 10, i32 1, i32 %11) ]
