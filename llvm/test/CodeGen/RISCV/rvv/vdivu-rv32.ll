; RUN: llc -mtriple=riscv32 -mattr=+experimental-v -verify-machineinstrs \
; RUN:   --riscv-no-aliases < %s | FileCheck %s
declare <vscale x 1 x i8> @llvm.riscv.vdivu.nxv1i8.nxv1i8(
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  i32);

define <vscale x 1 x i8> @intrinsic_vdivu_vv_nxv1i8_nxv1i8_nxv1i8(<vscale x 1 x i8> %0, <vscale x 1 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv1i8_nxv1i8_nxv1i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i8> @llvm.riscv.vdivu.nxv1i8.nxv1i8(
    <vscale x 1 x i8> %0,
    <vscale x 1 x i8> %1,
    i32 %2)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 1 x i8> @llvm.riscv.vdivu.mask.nxv1i8.nxv1i8(
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i8> @intrinsic_vdivu_mask_vv_nxv1i8_nxv1i8_nxv1i8(<vscale x 1 x i8> %0, <vscale x 1 x i8> %1, <vscale x 1 x i8> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv1i8_nxv1i8_nxv1i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i8> @llvm.riscv.vdivu.mask.nxv1i8.nxv1i8(
    <vscale x 1 x i8> %0,
    <vscale x 1 x i8> %1,
    <vscale x 1 x i8> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vdivu.nxv2i8.nxv2i8(
  <vscale x 2 x i8>,
  <vscale x 2 x i8>,
  i32);

define <vscale x 2 x i8> @intrinsic_vdivu_vv_nxv2i8_nxv2i8_nxv2i8(<vscale x 2 x i8> %0, <vscale x 2 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv2i8_nxv2i8_nxv2i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i8> @llvm.riscv.vdivu.nxv2i8.nxv2i8(
    <vscale x 2 x i8> %0,
    <vscale x 2 x i8> %1,
    i32 %2)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vdivu.mask.nxv2i8.nxv2i8(
  <vscale x 2 x i8>,
  <vscale x 2 x i8>,
  <vscale x 2 x i8>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i8> @intrinsic_vdivu_mask_vv_nxv2i8_nxv2i8_nxv2i8(<vscale x 2 x i8> %0, <vscale x 2 x i8> %1, <vscale x 2 x i8> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv2i8_nxv2i8_nxv2i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i8> @llvm.riscv.vdivu.mask.nxv2i8.nxv2i8(
    <vscale x 2 x i8> %0,
    <vscale x 2 x i8> %1,
    <vscale x 2 x i8> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vdivu.nxv4i8.nxv4i8(
  <vscale x 4 x i8>,
  <vscale x 4 x i8>,
  i32);

define <vscale x 4 x i8> @intrinsic_vdivu_vv_nxv4i8_nxv4i8_nxv4i8(<vscale x 4 x i8> %0, <vscale x 4 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv4i8_nxv4i8_nxv4i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i8> @llvm.riscv.vdivu.nxv4i8.nxv4i8(
    <vscale x 4 x i8> %0,
    <vscale x 4 x i8> %1,
    i32 %2)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vdivu.mask.nxv4i8.nxv4i8(
  <vscale x 4 x i8>,
  <vscale x 4 x i8>,
  <vscale x 4 x i8>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i8> @intrinsic_vdivu_mask_vv_nxv4i8_nxv4i8_nxv4i8(<vscale x 4 x i8> %0, <vscale x 4 x i8> %1, <vscale x 4 x i8> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv4i8_nxv4i8_nxv4i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i8> @llvm.riscv.vdivu.mask.nxv4i8.nxv4i8(
    <vscale x 4 x i8> %0,
    <vscale x 4 x i8> %1,
    <vscale x 4 x i8> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vdivu.nxv8i8.nxv8i8(
  <vscale x 8 x i8>,
  <vscale x 8 x i8>,
  i32);

define <vscale x 8 x i8> @intrinsic_vdivu_vv_nxv8i8_nxv8i8_nxv8i8(<vscale x 8 x i8> %0, <vscale x 8 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv8i8_nxv8i8_nxv8i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i8> @llvm.riscv.vdivu.nxv8i8.nxv8i8(
    <vscale x 8 x i8> %0,
    <vscale x 8 x i8> %1,
    i32 %2)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vdivu.mask.nxv8i8.nxv8i8(
  <vscale x 8 x i8>,
  <vscale x 8 x i8>,
  <vscale x 8 x i8>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i8> @intrinsic_vdivu_mask_vv_nxv8i8_nxv8i8_nxv8i8(<vscale x 8 x i8> %0, <vscale x 8 x i8> %1, <vscale x 8 x i8> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv8i8_nxv8i8_nxv8i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i8> @llvm.riscv.vdivu.mask.nxv8i8.nxv8i8(
    <vscale x 8 x i8> %0,
    <vscale x 8 x i8> %1,
    <vscale x 8 x i8> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vdivu.nxv16i8.nxv16i8(
  <vscale x 16 x i8>,
  <vscale x 16 x i8>,
  i32);

define <vscale x 16 x i8> @intrinsic_vdivu_vv_nxv16i8_nxv16i8_nxv16i8(<vscale x 16 x i8> %0, <vscale x 16 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv16i8_nxv16i8_nxv16i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i8> @llvm.riscv.vdivu.nxv16i8.nxv16i8(
    <vscale x 16 x i8> %0,
    <vscale x 16 x i8> %1,
    i32 %2)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vdivu.mask.nxv16i8.nxv16i8(
  <vscale x 16 x i8>,
  <vscale x 16 x i8>,
  <vscale x 16 x i8>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i8> @intrinsic_vdivu_mask_vv_nxv16i8_nxv16i8_nxv16i8(<vscale x 16 x i8> %0, <vscale x 16 x i8> %1, <vscale x 16 x i8> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv16i8_nxv16i8_nxv16i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i8> @llvm.riscv.vdivu.mask.nxv16i8.nxv16i8(
    <vscale x 16 x i8> %0,
    <vscale x 16 x i8> %1,
    <vscale x 16 x i8> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vdivu.nxv32i8.nxv32i8(
  <vscale x 32 x i8>,
  <vscale x 32 x i8>,
  i32);

define <vscale x 32 x i8> @intrinsic_vdivu_vv_nxv32i8_nxv32i8_nxv32i8(<vscale x 32 x i8> %0, <vscale x 32 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv32i8_nxv32i8_nxv32i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 32 x i8> @llvm.riscv.vdivu.nxv32i8.nxv32i8(
    <vscale x 32 x i8> %0,
    <vscale x 32 x i8> %1,
    i32 %2)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vdivu.mask.nxv32i8.nxv32i8(
  <vscale x 32 x i8>,
  <vscale x 32 x i8>,
  <vscale x 32 x i8>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i8> @intrinsic_vdivu_mask_vv_nxv32i8_nxv32i8_nxv32i8(<vscale x 32 x i8> %0, <vscale x 32 x i8> %1, <vscale x 32 x i8> %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv32i8_nxv32i8_nxv32i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i8> @llvm.riscv.vdivu.mask.nxv32i8.nxv32i8(
    <vscale x 32 x i8> %0,
    <vscale x 32 x i8> %1,
    <vscale x 32 x i8> %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 64 x i8> @llvm.riscv.vdivu.nxv64i8.nxv64i8(
  <vscale x 64 x i8>,
  <vscale x 64 x i8>,
  i32);

define <vscale x 64 x i8> @intrinsic_vdivu_vv_nxv64i8_nxv64i8_nxv64i8(<vscale x 64 x i8> %0, <vscale x 64 x i8> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv64i8_nxv64i8_nxv64i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 64 x i8> @llvm.riscv.vdivu.nxv64i8.nxv64i8(
    <vscale x 64 x i8> %0,
    <vscale x 64 x i8> %1,
    i32 %2)

  ret <vscale x 64 x i8> %a
}

declare <vscale x 64 x i8> @llvm.riscv.vdivu.mask.nxv64i8.nxv64i8(
  <vscale x 64 x i8>,
  <vscale x 64 x i8>,
  <vscale x 64 x i8>,
  <vscale x 64 x i1>,
  i32);

define <vscale x 64 x i8> @intrinsic_vdivu_mask_vv_nxv64i8_nxv64i8_nxv64i8(<vscale x 64 x i8> %0, <vscale x 64 x i8> %1, <vscale x 64 x i8> %2, <vscale x 64 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv64i8_nxv64i8_nxv64i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 64 x i8> @llvm.riscv.vdivu.mask.nxv64i8.nxv64i8(
    <vscale x 64 x i8> %0,
    <vscale x 64 x i8> %1,
    <vscale x 64 x i8> %2,
    <vscale x 64 x i1> %3,
    i32 %4)

  ret <vscale x 64 x i8> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vdivu.nxv1i16.nxv1i16(
  <vscale x 1 x i16>,
  <vscale x 1 x i16>,
  i32);

define <vscale x 1 x i16> @intrinsic_vdivu_vv_nxv1i16_nxv1i16_nxv1i16(<vscale x 1 x i16> %0, <vscale x 1 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv1i16_nxv1i16_nxv1i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i16> @llvm.riscv.vdivu.nxv1i16.nxv1i16(
    <vscale x 1 x i16> %0,
    <vscale x 1 x i16> %1,
    i32 %2)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vdivu.mask.nxv1i16.nxv1i16(
  <vscale x 1 x i16>,
  <vscale x 1 x i16>,
  <vscale x 1 x i16>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i16> @intrinsic_vdivu_mask_vv_nxv1i16_nxv1i16_nxv1i16(<vscale x 1 x i16> %0, <vscale x 1 x i16> %1, <vscale x 1 x i16> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv1i16_nxv1i16_nxv1i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i16> @llvm.riscv.vdivu.mask.nxv1i16.nxv1i16(
    <vscale x 1 x i16> %0,
    <vscale x 1 x i16> %1,
    <vscale x 1 x i16> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vdivu.nxv2i16.nxv2i16(
  <vscale x 2 x i16>,
  <vscale x 2 x i16>,
  i32);

define <vscale x 2 x i16> @intrinsic_vdivu_vv_nxv2i16_nxv2i16_nxv2i16(<vscale x 2 x i16> %0, <vscale x 2 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv2i16_nxv2i16_nxv2i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i16> @llvm.riscv.vdivu.nxv2i16.nxv2i16(
    <vscale x 2 x i16> %0,
    <vscale x 2 x i16> %1,
    i32 %2)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vdivu.mask.nxv2i16.nxv2i16(
  <vscale x 2 x i16>,
  <vscale x 2 x i16>,
  <vscale x 2 x i16>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i16> @intrinsic_vdivu_mask_vv_nxv2i16_nxv2i16_nxv2i16(<vscale x 2 x i16> %0, <vscale x 2 x i16> %1, <vscale x 2 x i16> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv2i16_nxv2i16_nxv2i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i16> @llvm.riscv.vdivu.mask.nxv2i16.nxv2i16(
    <vscale x 2 x i16> %0,
    <vscale x 2 x i16> %1,
    <vscale x 2 x i16> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vdivu.nxv4i16.nxv4i16(
  <vscale x 4 x i16>,
  <vscale x 4 x i16>,
  i32);

define <vscale x 4 x i16> @intrinsic_vdivu_vv_nxv4i16_nxv4i16_nxv4i16(<vscale x 4 x i16> %0, <vscale x 4 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv4i16_nxv4i16_nxv4i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i16> @llvm.riscv.vdivu.nxv4i16.nxv4i16(
    <vscale x 4 x i16> %0,
    <vscale x 4 x i16> %1,
    i32 %2)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vdivu.mask.nxv4i16.nxv4i16(
  <vscale x 4 x i16>,
  <vscale x 4 x i16>,
  <vscale x 4 x i16>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i16> @intrinsic_vdivu_mask_vv_nxv4i16_nxv4i16_nxv4i16(<vscale x 4 x i16> %0, <vscale x 4 x i16> %1, <vscale x 4 x i16> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv4i16_nxv4i16_nxv4i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i16> @llvm.riscv.vdivu.mask.nxv4i16.nxv4i16(
    <vscale x 4 x i16> %0,
    <vscale x 4 x i16> %1,
    <vscale x 4 x i16> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vdivu.nxv8i16.nxv8i16(
  <vscale x 8 x i16>,
  <vscale x 8 x i16>,
  i32);

define <vscale x 8 x i16> @intrinsic_vdivu_vv_nxv8i16_nxv8i16_nxv8i16(<vscale x 8 x i16> %0, <vscale x 8 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv8i16_nxv8i16_nxv8i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i16> @llvm.riscv.vdivu.nxv8i16.nxv8i16(
    <vscale x 8 x i16> %0,
    <vscale x 8 x i16> %1,
    i32 %2)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vdivu.mask.nxv8i16.nxv8i16(
  <vscale x 8 x i16>,
  <vscale x 8 x i16>,
  <vscale x 8 x i16>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i16> @intrinsic_vdivu_mask_vv_nxv8i16_nxv8i16_nxv8i16(<vscale x 8 x i16> %0, <vscale x 8 x i16> %1, <vscale x 8 x i16> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv8i16_nxv8i16_nxv8i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i16> @llvm.riscv.vdivu.mask.nxv8i16.nxv8i16(
    <vscale x 8 x i16> %0,
    <vscale x 8 x i16> %1,
    <vscale x 8 x i16> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vdivu.nxv16i16.nxv16i16(
  <vscale x 16 x i16>,
  <vscale x 16 x i16>,
  i32);

define <vscale x 16 x i16> @intrinsic_vdivu_vv_nxv16i16_nxv16i16_nxv16i16(<vscale x 16 x i16> %0, <vscale x 16 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv16i16_nxv16i16_nxv16i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i16> @llvm.riscv.vdivu.nxv16i16.nxv16i16(
    <vscale x 16 x i16> %0,
    <vscale x 16 x i16> %1,
    i32 %2)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vdivu.mask.nxv16i16.nxv16i16(
  <vscale x 16 x i16>,
  <vscale x 16 x i16>,
  <vscale x 16 x i16>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i16> @intrinsic_vdivu_mask_vv_nxv16i16_nxv16i16_nxv16i16(<vscale x 16 x i16> %0, <vscale x 16 x i16> %1, <vscale x 16 x i16> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv16i16_nxv16i16_nxv16i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i16> @llvm.riscv.vdivu.mask.nxv16i16.nxv16i16(
    <vscale x 16 x i16> %0,
    <vscale x 16 x i16> %1,
    <vscale x 16 x i16> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vdivu.nxv32i16.nxv32i16(
  <vscale x 32 x i16>,
  <vscale x 32 x i16>,
  i32);

define <vscale x 32 x i16> @intrinsic_vdivu_vv_nxv32i16_nxv32i16_nxv32i16(<vscale x 32 x i16> %0, <vscale x 32 x i16> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv32i16_nxv32i16_nxv32i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 32 x i16> @llvm.riscv.vdivu.nxv32i16.nxv32i16(
    <vscale x 32 x i16> %0,
    <vscale x 32 x i16> %1,
    i32 %2)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vdivu.mask.nxv32i16.nxv32i16(
  <vscale x 32 x i16>,
  <vscale x 32 x i16>,
  <vscale x 32 x i16>,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i16> @intrinsic_vdivu_mask_vv_nxv32i16_nxv32i16_nxv32i16(<vscale x 32 x i16> %0, <vscale x 32 x i16> %1, <vscale x 32 x i16> %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv32i16_nxv32i16_nxv32i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 32 x i16> @llvm.riscv.vdivu.mask.nxv32i16.nxv32i16(
    <vscale x 32 x i16> %0,
    <vscale x 32 x i16> %1,
    <vscale x 32 x i16> %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vdivu.nxv1i32.nxv1i32(
  <vscale x 1 x i32>,
  <vscale x 1 x i32>,
  i32);

define <vscale x 1 x i32> @intrinsic_vdivu_vv_nxv1i32_nxv1i32_nxv1i32(<vscale x 1 x i32> %0, <vscale x 1 x i32> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv1i32_nxv1i32_nxv1i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 1 x i32> @llvm.riscv.vdivu.nxv1i32.nxv1i32(
    <vscale x 1 x i32> %0,
    <vscale x 1 x i32> %1,
    i32 %2)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vdivu.mask.nxv1i32.nxv1i32(
  <vscale x 1 x i32>,
  <vscale x 1 x i32>,
  <vscale x 1 x i32>,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i32> @intrinsic_vdivu_mask_vv_nxv1i32_nxv1i32_nxv1i32(<vscale x 1 x i32> %0, <vscale x 1 x i32> %1, <vscale x 1 x i32> %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv1i32_nxv1i32_nxv1i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 1 x i32> @llvm.riscv.vdivu.mask.nxv1i32.nxv1i32(
    <vscale x 1 x i32> %0,
    <vscale x 1 x i32> %1,
    <vscale x 1 x i32> %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vdivu.nxv2i32.nxv2i32(
  <vscale x 2 x i32>,
  <vscale x 2 x i32>,
  i32);

define <vscale x 2 x i32> @intrinsic_vdivu_vv_nxv2i32_nxv2i32_nxv2i32(<vscale x 2 x i32> %0, <vscale x 2 x i32> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv2i32_nxv2i32_nxv2i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 2 x i32> @llvm.riscv.vdivu.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %0,
    <vscale x 2 x i32> %1,
    i32 %2)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vdivu.mask.nxv2i32.nxv2i32(
  <vscale x 2 x i32>,
  <vscale x 2 x i32>,
  <vscale x 2 x i32>,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i32> @intrinsic_vdivu_mask_vv_nxv2i32_nxv2i32_nxv2i32(<vscale x 2 x i32> %0, <vscale x 2 x i32> %1, <vscale x 2 x i32> %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv2i32_nxv2i32_nxv2i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 2 x i32> @llvm.riscv.vdivu.mask.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %0,
    <vscale x 2 x i32> %1,
    <vscale x 2 x i32> %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vdivu.nxv4i32.nxv4i32(
  <vscale x 4 x i32>,
  <vscale x 4 x i32>,
  i32);

define <vscale x 4 x i32> @intrinsic_vdivu_vv_nxv4i32_nxv4i32_nxv4i32(<vscale x 4 x i32> %0, <vscale x 4 x i32> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv4i32_nxv4i32_nxv4i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 4 x i32> @llvm.riscv.vdivu.nxv4i32.nxv4i32(
    <vscale x 4 x i32> %0,
    <vscale x 4 x i32> %1,
    i32 %2)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vdivu.mask.nxv4i32.nxv4i32(
  <vscale x 4 x i32>,
  <vscale x 4 x i32>,
  <vscale x 4 x i32>,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i32> @intrinsic_vdivu_mask_vv_nxv4i32_nxv4i32_nxv4i32(<vscale x 4 x i32> %0, <vscale x 4 x i32> %1, <vscale x 4 x i32> %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv4i32_nxv4i32_nxv4i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 4 x i32> @llvm.riscv.vdivu.mask.nxv4i32.nxv4i32(
    <vscale x 4 x i32> %0,
    <vscale x 4 x i32> %1,
    <vscale x 4 x i32> %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vdivu.nxv8i32.nxv8i32(
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  i32);

define <vscale x 8 x i32> @intrinsic_vdivu_vv_nxv8i32_nxv8i32_nxv8i32(<vscale x 8 x i32> %0, <vscale x 8 x i32> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv8i32_nxv8i32_nxv8i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 8 x i32> @llvm.riscv.vdivu.nxv8i32.nxv8i32(
    <vscale x 8 x i32> %0,
    <vscale x 8 x i32> %1,
    i32 %2)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vdivu.mask.nxv8i32.nxv8i32(
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i32> @intrinsic_vdivu_mask_vv_nxv8i32_nxv8i32_nxv8i32(<vscale x 8 x i32> %0, <vscale x 8 x i32> %1, <vscale x 8 x i32> %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv8i32_nxv8i32_nxv8i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 8 x i32> @llvm.riscv.vdivu.mask.nxv8i32.nxv8i32(
    <vscale x 8 x i32> %0,
    <vscale x 8 x i32> %1,
    <vscale x 8 x i32> %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vdivu.nxv16i32.nxv16i32(
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  i32);

define <vscale x 16 x i32> @intrinsic_vdivu_vv_nxv16i32_nxv16i32_nxv16i32(<vscale x 16 x i32> %0, <vscale x 16 x i32> %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vv_nxv16i32_nxv16i32_nxv16i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
  %a = call <vscale x 16 x i32> @llvm.riscv.vdivu.nxv16i32.nxv16i32(
    <vscale x 16 x i32> %0,
    <vscale x 16 x i32> %1,
    i32 %2)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vdivu.mask.nxv16i32.nxv16i32(
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i32> @intrinsic_vdivu_mask_vv_nxv16i32_nxv16i32_nxv16i32(<vscale x 16 x i32> %0, <vscale x 16 x i32> %1, <vscale x 16 x i32> %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vv_nxv16i32_nxv16i32_nxv16i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vdivu.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}, v0.t
  %a = call <vscale x 16 x i32> @llvm.riscv.vdivu.mask.nxv16i32.nxv16i32(
    <vscale x 16 x i32> %0,
    <vscale x 16 x i32> %1,
    <vscale x 16 x i32> %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 1 x i8> @llvm.riscv.vdivu.nxv1i8.i8(
  <vscale x 1 x i8>,
  i8,
  i32);

define <vscale x 1 x i8> @intrinsic_vdivu_vx_nxv1i8_nxv1i8_i8(<vscale x 1 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv1i8_nxv1i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 1 x i8> @llvm.riscv.vdivu.nxv1i8.i8(
    <vscale x 1 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 1 x i8> @llvm.riscv.vdivu.mask.nxv1i8.i8(
  <vscale x 1 x i8>,
  <vscale x 1 x i8>,
  i8,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i8> @intrinsic_vdivu_mask_vx_nxv1i8_nxv1i8_i8(<vscale x 1 x i8> %0, <vscale x 1 x i8> %1, i8 %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv1i8_nxv1i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 1 x i8> @llvm.riscv.vdivu.mask.nxv1i8.i8(
    <vscale x 1 x i8> %0,
    <vscale x 1 x i8> %1,
    i8 %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vdivu.nxv2i8.i8(
  <vscale x 2 x i8>,
  i8,
  i32);

define <vscale x 2 x i8> @intrinsic_vdivu_vx_nxv2i8_nxv2i8_i8(<vscale x 2 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv2i8_nxv2i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 2 x i8> @llvm.riscv.vdivu.nxv2i8.i8(
    <vscale x 2 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 2 x i8> @llvm.riscv.vdivu.mask.nxv2i8.i8(
  <vscale x 2 x i8>,
  <vscale x 2 x i8>,
  i8,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i8> @intrinsic_vdivu_mask_vx_nxv2i8_nxv2i8_i8(<vscale x 2 x i8> %0, <vscale x 2 x i8> %1, i8 %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv2i8_nxv2i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 2 x i8> @llvm.riscv.vdivu.mask.nxv2i8.i8(
    <vscale x 2 x i8> %0,
    <vscale x 2 x i8> %1,
    i8 %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vdivu.nxv4i8.i8(
  <vscale x 4 x i8>,
  i8,
  i32);

define <vscale x 4 x i8> @intrinsic_vdivu_vx_nxv4i8_nxv4i8_i8(<vscale x 4 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv4i8_nxv4i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 4 x i8> @llvm.riscv.vdivu.nxv4i8.i8(
    <vscale x 4 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 4 x i8> @llvm.riscv.vdivu.mask.nxv4i8.i8(
  <vscale x 4 x i8>,
  <vscale x 4 x i8>,
  i8,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i8> @intrinsic_vdivu_mask_vx_nxv4i8_nxv4i8_i8(<vscale x 4 x i8> %0, <vscale x 4 x i8> %1, i8 %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv4i8_nxv4i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 4 x i8> @llvm.riscv.vdivu.mask.nxv4i8.i8(
    <vscale x 4 x i8> %0,
    <vscale x 4 x i8> %1,
    i8 %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vdivu.nxv8i8.i8(
  <vscale x 8 x i8>,
  i8,
  i32);

define <vscale x 8 x i8> @intrinsic_vdivu_vx_nxv8i8_nxv8i8_i8(<vscale x 8 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv8i8_nxv8i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 8 x i8> @llvm.riscv.vdivu.nxv8i8.i8(
    <vscale x 8 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 8 x i8> @llvm.riscv.vdivu.mask.nxv8i8.i8(
  <vscale x 8 x i8>,
  <vscale x 8 x i8>,
  i8,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i8> @intrinsic_vdivu_mask_vx_nxv8i8_nxv8i8_i8(<vscale x 8 x i8> %0, <vscale x 8 x i8> %1, i8 %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv8i8_nxv8i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 8 x i8> @llvm.riscv.vdivu.mask.nxv8i8.i8(
    <vscale x 8 x i8> %0,
    <vscale x 8 x i8> %1,
    i8 %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vdivu.nxv16i8.i8(
  <vscale x 16 x i8>,
  i8,
  i32);

define <vscale x 16 x i8> @intrinsic_vdivu_vx_nxv16i8_nxv16i8_i8(<vscale x 16 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv16i8_nxv16i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 16 x i8> @llvm.riscv.vdivu.nxv16i8.i8(
    <vscale x 16 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 16 x i8> @llvm.riscv.vdivu.mask.nxv16i8.i8(
  <vscale x 16 x i8>,
  <vscale x 16 x i8>,
  i8,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i8> @intrinsic_vdivu_mask_vx_nxv16i8_nxv16i8_i8(<vscale x 16 x i8> %0, <vscale x 16 x i8> %1, i8 %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv16i8_nxv16i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 16 x i8> @llvm.riscv.vdivu.mask.nxv16i8.i8(
    <vscale x 16 x i8> %0,
    <vscale x 16 x i8> %1,
    i8 %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vdivu.nxv32i8.i8(
  <vscale x 32 x i8>,
  i8,
  i32);

define <vscale x 32 x i8> @intrinsic_vdivu_vx_nxv32i8_nxv32i8_i8(<vscale x 32 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv32i8_nxv32i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 32 x i8> @llvm.riscv.vdivu.nxv32i8.i8(
    <vscale x 32 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 32 x i8> @llvm.riscv.vdivu.mask.nxv32i8.i8(
  <vscale x 32 x i8>,
  <vscale x 32 x i8>,
  i8,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i8> @intrinsic_vdivu_mask_vx_nxv32i8_nxv32i8_i8(<vscale x 32 x i8> %0, <vscale x 32 x i8> %1, i8 %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv32i8_nxv32i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 32 x i8> @llvm.riscv.vdivu.mask.nxv32i8.i8(
    <vscale x 32 x i8> %0,
    <vscale x 32 x i8> %1,
    i8 %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 32 x i8> %a
}

declare <vscale x 64 x i8> @llvm.riscv.vdivu.nxv64i8.i8(
  <vscale x 64 x i8>,
  i8,
  i32);

define <vscale x 64 x i8> @intrinsic_vdivu_vx_nxv64i8_nxv64i8_i8(<vscale x 64 x i8> %0, i8 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv64i8_nxv64i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 64 x i8> @llvm.riscv.vdivu.nxv64i8.i8(
    <vscale x 64 x i8> %0,
    i8 %1,
    i32 %2)

  ret <vscale x 64 x i8> %a
}

declare <vscale x 64 x i8> @llvm.riscv.vdivu.mask.nxv64i8.i8(
  <vscale x 64 x i8>,
  <vscale x 64 x i8>,
  i8,
  <vscale x 64 x i1>,
  i32);

define <vscale x 64 x i8> @intrinsic_vdivu_mask_vx_nxv64i8_nxv64i8_i8(<vscale x 64 x i8> %0, <vscale x 64 x i8> %1, i8 %2, <vscale x 64 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv64i8_nxv64i8_i8
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e8,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 64 x i8> @llvm.riscv.vdivu.mask.nxv64i8.i8(
    <vscale x 64 x i8> %0,
    <vscale x 64 x i8> %1,
    i8 %2,
    <vscale x 64 x i1> %3,
    i32 %4)

  ret <vscale x 64 x i8> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vdivu.nxv1i16.i16(
  <vscale x 1 x i16>,
  i16,
  i32);

define <vscale x 1 x i16> @intrinsic_vdivu_vx_nxv1i16_nxv1i16_i16(<vscale x 1 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv1i16_nxv1i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 1 x i16> @llvm.riscv.vdivu.nxv1i16.i16(
    <vscale x 1 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 1 x i16> @llvm.riscv.vdivu.mask.nxv1i16.i16(
  <vscale x 1 x i16>,
  <vscale x 1 x i16>,
  i16,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i16> @intrinsic_vdivu_mask_vx_nxv1i16_nxv1i16_i16(<vscale x 1 x i16> %0, <vscale x 1 x i16> %1, i16 %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv1i16_nxv1i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 1 x i16> @llvm.riscv.vdivu.mask.nxv1i16.i16(
    <vscale x 1 x i16> %0,
    <vscale x 1 x i16> %1,
    i16 %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vdivu.nxv2i16.i16(
  <vscale x 2 x i16>,
  i16,
  i32);

define <vscale x 2 x i16> @intrinsic_vdivu_vx_nxv2i16_nxv2i16_i16(<vscale x 2 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv2i16_nxv2i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 2 x i16> @llvm.riscv.vdivu.nxv2i16.i16(
    <vscale x 2 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 2 x i16> @llvm.riscv.vdivu.mask.nxv2i16.i16(
  <vscale x 2 x i16>,
  <vscale x 2 x i16>,
  i16,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i16> @intrinsic_vdivu_mask_vx_nxv2i16_nxv2i16_i16(<vscale x 2 x i16> %0, <vscale x 2 x i16> %1, i16 %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv2i16_nxv2i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 2 x i16> @llvm.riscv.vdivu.mask.nxv2i16.i16(
    <vscale x 2 x i16> %0,
    <vscale x 2 x i16> %1,
    i16 %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vdivu.nxv4i16.i16(
  <vscale x 4 x i16>,
  i16,
  i32);

define <vscale x 4 x i16> @intrinsic_vdivu_vx_nxv4i16_nxv4i16_i16(<vscale x 4 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv4i16_nxv4i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 4 x i16> @llvm.riscv.vdivu.nxv4i16.i16(
    <vscale x 4 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 4 x i16> @llvm.riscv.vdivu.mask.nxv4i16.i16(
  <vscale x 4 x i16>,
  <vscale x 4 x i16>,
  i16,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i16> @intrinsic_vdivu_mask_vx_nxv4i16_nxv4i16_i16(<vscale x 4 x i16> %0, <vscale x 4 x i16> %1, i16 %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv4i16_nxv4i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 4 x i16> @llvm.riscv.vdivu.mask.nxv4i16.i16(
    <vscale x 4 x i16> %0,
    <vscale x 4 x i16> %1,
    i16 %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vdivu.nxv8i16.i16(
  <vscale x 8 x i16>,
  i16,
  i32);

define <vscale x 8 x i16> @intrinsic_vdivu_vx_nxv8i16_nxv8i16_i16(<vscale x 8 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv8i16_nxv8i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 8 x i16> @llvm.riscv.vdivu.nxv8i16.i16(
    <vscale x 8 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 8 x i16> @llvm.riscv.vdivu.mask.nxv8i16.i16(
  <vscale x 8 x i16>,
  <vscale x 8 x i16>,
  i16,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i16> @intrinsic_vdivu_mask_vx_nxv8i16_nxv8i16_i16(<vscale x 8 x i16> %0, <vscale x 8 x i16> %1, i16 %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv8i16_nxv8i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 8 x i16> @llvm.riscv.vdivu.mask.nxv8i16.i16(
    <vscale x 8 x i16> %0,
    <vscale x 8 x i16> %1,
    i16 %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vdivu.nxv16i16.i16(
  <vscale x 16 x i16>,
  i16,
  i32);

define <vscale x 16 x i16> @intrinsic_vdivu_vx_nxv16i16_nxv16i16_i16(<vscale x 16 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv16i16_nxv16i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 16 x i16> @llvm.riscv.vdivu.nxv16i16.i16(
    <vscale x 16 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 16 x i16> @llvm.riscv.vdivu.mask.nxv16i16.i16(
  <vscale x 16 x i16>,
  <vscale x 16 x i16>,
  i16,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i16> @intrinsic_vdivu_mask_vx_nxv16i16_nxv16i16_i16(<vscale x 16 x i16> %0, <vscale x 16 x i16> %1, i16 %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv16i16_nxv16i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 16 x i16> @llvm.riscv.vdivu.mask.nxv16i16.i16(
    <vscale x 16 x i16> %0,
    <vscale x 16 x i16> %1,
    i16 %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vdivu.nxv32i16.i16(
  <vscale x 32 x i16>,
  i16,
  i32);

define <vscale x 32 x i16> @intrinsic_vdivu_vx_nxv32i16_nxv32i16_i16(<vscale x 32 x i16> %0, i16 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv32i16_nxv32i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 32 x i16> @llvm.riscv.vdivu.nxv32i16.i16(
    <vscale x 32 x i16> %0,
    i16 %1,
    i32 %2)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 32 x i16> @llvm.riscv.vdivu.mask.nxv32i16.i16(
  <vscale x 32 x i16>,
  <vscale x 32 x i16>,
  i16,
  <vscale x 32 x i1>,
  i32);

define <vscale x 32 x i16> @intrinsic_vdivu_mask_vx_nxv32i16_nxv32i16_i16(<vscale x 32 x i16> %0, <vscale x 32 x i16> %1, i16 %2, <vscale x 32 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv32i16_nxv32i16_i16
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e16,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 32 x i16> @llvm.riscv.vdivu.mask.nxv32i16.i16(
    <vscale x 32 x i16> %0,
    <vscale x 32 x i16> %1,
    i16 %2,
    <vscale x 32 x i1> %3,
    i32 %4)

  ret <vscale x 32 x i16> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vdivu.nxv1i32.i32(
  <vscale x 1 x i32>,
  i32,
  i32);

define <vscale x 1 x i32> @intrinsic_vdivu_vx_nxv1i32_nxv1i32_i32(<vscale x 1 x i32> %0, i32 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv1i32_nxv1i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 1 x i32> @llvm.riscv.vdivu.nxv1i32.i32(
    <vscale x 1 x i32> %0,
    i32 %1,
    i32 %2)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 1 x i32> @llvm.riscv.vdivu.mask.nxv1i32.i32(
  <vscale x 1 x i32>,
  <vscale x 1 x i32>,
  i32,
  <vscale x 1 x i1>,
  i32);

define <vscale x 1 x i32> @intrinsic_vdivu_mask_vx_nxv1i32_nxv1i32_i32(<vscale x 1 x i32> %0, <vscale x 1 x i32> %1, i32 %2, <vscale x 1 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv1i32_nxv1i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,mf2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 1 x i32> @llvm.riscv.vdivu.mask.nxv1i32.i32(
    <vscale x 1 x i32> %0,
    <vscale x 1 x i32> %1,
    i32 %2,
    <vscale x 1 x i1> %3,
    i32 %4)

  ret <vscale x 1 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vdivu.nxv2i32.i32(
  <vscale x 2 x i32>,
  i32,
  i32);

define <vscale x 2 x i32> @intrinsic_vdivu_vx_nxv2i32_nxv2i32_i32(<vscale x 2 x i32> %0, i32 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv2i32_nxv2i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 2 x i32> @llvm.riscv.vdivu.nxv2i32.i32(
    <vscale x 2 x i32> %0,
    i32 %1,
    i32 %2)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 2 x i32> @llvm.riscv.vdivu.mask.nxv2i32.i32(
  <vscale x 2 x i32>,
  <vscale x 2 x i32>,
  i32,
  <vscale x 2 x i1>,
  i32);

define <vscale x 2 x i32> @intrinsic_vdivu_mask_vx_nxv2i32_nxv2i32_i32(<vscale x 2 x i32> %0, <vscale x 2 x i32> %1, i32 %2, <vscale x 2 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv2i32_nxv2i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m1,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 2 x i32> @llvm.riscv.vdivu.mask.nxv2i32.i32(
    <vscale x 2 x i32> %0,
    <vscale x 2 x i32> %1,
    i32 %2,
    <vscale x 2 x i1> %3,
    i32 %4)

  ret <vscale x 2 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vdivu.nxv4i32.i32(
  <vscale x 4 x i32>,
  i32,
  i32);

define <vscale x 4 x i32> @intrinsic_vdivu_vx_nxv4i32_nxv4i32_i32(<vscale x 4 x i32> %0, i32 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv4i32_nxv4i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 4 x i32> @llvm.riscv.vdivu.nxv4i32.i32(
    <vscale x 4 x i32> %0,
    i32 %1,
    i32 %2)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 4 x i32> @llvm.riscv.vdivu.mask.nxv4i32.i32(
  <vscale x 4 x i32>,
  <vscale x 4 x i32>,
  i32,
  <vscale x 4 x i1>,
  i32);

define <vscale x 4 x i32> @intrinsic_vdivu_mask_vx_nxv4i32_nxv4i32_i32(<vscale x 4 x i32> %0, <vscale x 4 x i32> %1, i32 %2, <vscale x 4 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv4i32_nxv4i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m2,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 4 x i32> @llvm.riscv.vdivu.mask.nxv4i32.i32(
    <vscale x 4 x i32> %0,
    <vscale x 4 x i32> %1,
    i32 %2,
    <vscale x 4 x i1> %3,
    i32 %4)

  ret <vscale x 4 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vdivu.nxv8i32.i32(
  <vscale x 8 x i32>,
  i32,
  i32);

define <vscale x 8 x i32> @intrinsic_vdivu_vx_nxv8i32_nxv8i32_i32(<vscale x 8 x i32> %0, i32 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv8i32_nxv8i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 8 x i32> @llvm.riscv.vdivu.nxv8i32.i32(
    <vscale x 8 x i32> %0,
    i32 %1,
    i32 %2)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 8 x i32> @llvm.riscv.vdivu.mask.nxv8i32.i32(
  <vscale x 8 x i32>,
  <vscale x 8 x i32>,
  i32,
  <vscale x 8 x i1>,
  i32);

define <vscale x 8 x i32> @intrinsic_vdivu_mask_vx_nxv8i32_nxv8i32_i32(<vscale x 8 x i32> %0, <vscale x 8 x i32> %1, i32 %2, <vscale x 8 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv8i32_nxv8i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m4,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 8 x i32> @llvm.riscv.vdivu.mask.nxv8i32.i32(
    <vscale x 8 x i32> %0,
    <vscale x 8 x i32> %1,
    i32 %2,
    <vscale x 8 x i1> %3,
    i32 %4)

  ret <vscale x 8 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vdivu.nxv16i32.i32(
  <vscale x 16 x i32>,
  i32,
  i32);

define <vscale x 16 x i32> @intrinsic_vdivu_vx_nxv16i32_nxv16i32_i32(<vscale x 16 x i32> %0, i32 %1, i32 %2) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_vx_nxv16i32_nxv16i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}
  %a = call <vscale x 16 x i32> @llvm.riscv.vdivu.nxv16i32.i32(
    <vscale x 16 x i32> %0,
    i32 %1,
    i32 %2)

  ret <vscale x 16 x i32> %a
}

declare <vscale x 16 x i32> @llvm.riscv.vdivu.mask.nxv16i32.i32(
  <vscale x 16 x i32>,
  <vscale x 16 x i32>,
  i32,
  <vscale x 16 x i1>,
  i32);

define <vscale x 16 x i32> @intrinsic_vdivu_mask_vx_nxv16i32_nxv16i32_i32(<vscale x 16 x i32> %0, <vscale x 16 x i32> %1, i32 %2, <vscale x 16 x i1> %3, i32 %4) nounwind {
entry:
; CHECK-LABEL: intrinsic_vdivu_mask_vx_nxv16i32_nxv16i32_i32
; CHECK:       vsetvli {{.*}}, {{a[0-9]+}}, e32,m8,ta,mu
; CHECK:       vdivu.vx {{v[0-9]+}}, {{v[0-9]+}}, {{a[0-9]+}}, v0.t
  %a = call <vscale x 16 x i32> @llvm.riscv.vdivu.mask.nxv16i32.i32(
    <vscale x 16 x i32> %0,
    <vscale x 16 x i32> %1,
    i32 %2,
    <vscale x 16 x i1> %3,
    i32 %4)

  ret <vscale x 16 x i32> %a
}
