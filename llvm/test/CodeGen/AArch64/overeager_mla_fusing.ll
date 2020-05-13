; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc %s --mtriple aarch64 -verify-machineinstrs -o - | FileCheck %s

define dso_local void @jsimd_idct_ifast_neon_intrinsic(i8* nocapture readonly %dct_table, i16* nocapture readonly %coef_block, i8** nocapture readonly %output_buf, i32 %output_col) local_unnamed_addr #0 {
; CHECK-LABEL: jsimd_idct_ifast_neon_intrinsic:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    ldr q0, [x1, #32]
; CHECK-NEXT:    ldr q1, [x1, #96]
; CHECK-NEXT:    ldr q2, [x0, #32]
; CHECK-NEXT:    ldr q3, [x0, #96]
; CHECK-NEXT:    ldr x8, [x2, #48]
; CHECK-NEXT:    mov w9, w3
; CHECK-NEXT:    mul v0.8h, v2.8h, v0.8h
; CHECK-NEXT:    mul v1.8h, v3.8h, v1.8h
; CHECK-NEXT:    add v2.8h, v0.8h, v1.8h
; CHECK-NEXT:    str q2, [x8, x9]
; CHECK-NEXT:    ldr x8, [x2, #56]
; CHECK-NEXT:    sub v0.8h, v0.8h, v1.8h
; CHECK-NEXT:    str q0, [x8, x9]
; CHECK-NEXT:    ret
entry:
  %add.ptr5 = getelementptr inbounds i16, i16* %coef_block, i64 16
  %0 = bitcast i16* %add.ptr5 to <8 x i16>*
  %1 = load <8 x i16>, <8 x i16>* %0, align 16

  %add.ptr17 = getelementptr inbounds i16, i16* %coef_block, i64 48
  %2 = bitcast i16* %add.ptr17 to <8 x i16>*
  %3 = load <8 x i16>, <8 x i16>* %2, align 16

  %add.ptr29 = getelementptr inbounds i8, i8* %dct_table, i64 32
  %4 = bitcast i8* %add.ptr29 to <8 x i16>*
  %5 = load <8 x i16>, <8 x i16>* %4, align 16

  %add.ptr41 = getelementptr inbounds i8, i8* %dct_table, i64 96
  %6 = bitcast i8* %add.ptr41 to <8 x i16>*
  %7 = load <8 x i16>, <8 x i16>* %6, align 16

  %mul.i966 = mul <8 x i16> %5, %1
  %mul.i964 = mul <8 x i16> %7, %3

  %add.i961 = add <8 x i16> %mul.i966, %mul.i964
  %sub.i960 = sub <8 x i16> %mul.i966, %mul.i964

  %idx.ext = zext i32 %output_col to i64

  %arrayidx404 = getelementptr inbounds i8*, i8** %output_buf, i64 6
  %8 = load i8*, i8** %arrayidx404, align 8
  %add.ptr406 = getelementptr inbounds i8, i8* %8, i64 %idx.ext
  %9 = bitcast i8* %add.ptr406 to <8 x i16>*
  store <8 x i16> %add.i961, <8 x i16>* %9, align 8

  %arrayidx408 = getelementptr inbounds i8*, i8** %output_buf, i64 7
  %10 = load i8*, i8** %arrayidx408, align 8
  %add.ptr410 = getelementptr inbounds i8, i8* %10, i64 %idx.ext
  %11 = bitcast i8* %add.ptr410 to <8 x i16>*
  store <8 x i16> %sub.i960, <8 x i16>* %11, align 8

  ret void
}
