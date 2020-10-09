// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+WavefrontSize32,-WavefrontSize64 -show-encoding %s | FileCheck --check-prefixes=GFX10,W32 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-WavefrontSize32,+WavefrontSize64 -show-encoding %s | FileCheck --check-prefixes=GFX10,W64 %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=+WavefrontSize32,-WavefrontSize64 %s 2>&1 | FileCheck --check-prefixes=GFX10-ERR,W32-ERR --implicit-check-not=error: %s
// RUN: not llvm-mc -arch=amdgcn -mcpu=gfx1010 -mattr=-WavefrontSize32,+WavefrontSize64 %s 2>&1 | FileCheck --check-prefixes=GFX10-ERR,W64-ERR --implicit-check-not=error: %s

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x02,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_i32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x0a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_u32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x0c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_u32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x0e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x10,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x14,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x16,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_rpi_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x18,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_flr_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x1a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_off_f32_i4_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x1c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte0_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x22,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte1_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x24,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte2_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x26,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte3_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x28,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_fract_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x40,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_trunc_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x42,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ceil_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x44,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rndne_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x46,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_floor_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x48,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_exp_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x4a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_log_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x4e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x54,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_iflag_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x56,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rsq_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x5c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sqrt_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x66,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sin_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x6a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cos_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x6c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_not_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x6e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_bfrev_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x70,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbh_u32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x72,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbl_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x74,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbh_i32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x76,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_exp_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x7e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_mant_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x80,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_u16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xa0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_i16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xa2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_u16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xa4,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xa6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xa8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sqrt_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xaa,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rsq_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xac,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_log_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xae,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_exp_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xb0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_mant_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xb2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_exp_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xb4,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_floor_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xb6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ceil_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xb8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_trunc_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xba,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rndne_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xbc,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_fract_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xbe,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sin_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xc0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cos_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xc2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_norm_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xc6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_norm_u16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0xc8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_add_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x06,0x01,0x88,0xc6,0xfa]

v_sub_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x08,0x01,0x88,0xc6,0xfa]

v_subrev_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x0a,0x01,0x88,0xc6,0xfa]

v_mul_legacy_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x0e,0x01,0x88,0xc6,0xfa]

v_mul_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x10,0x01,0x88,0xc6,0xfa]

v_mul_i32_i24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x12,0x01,0x88,0xc6,0xfa]

v_mul_hi_i32_i24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x14,0x01,0x88,0xc6,0xfa]

v_mul_u32_u24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x16,0x01,0x88,0xc6,0xfa]

v_mul_hi_u32_u24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x18,0x01,0x88,0xc6,0xfa]

v_min_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x1e,0x01,0x88,0xc6,0xfa]

v_max_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x20,0x01,0x88,0xc6,0xfa]

v_min_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x22,0x01,0x88,0xc6,0xfa]

v_max_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x24,0x01,0x88,0xc6,0xfa]

v_min_u32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x26,0x01,0x88,0xc6,0xfa]

v_max_u32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x28,0x01,0x88,0xc6,0xfa]

v_lshrrev_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x2c,0x01,0x88,0xc6,0xfa]

v_ashrrev_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x30,0x01,0x88,0xc6,0xfa]

v_lshlrev_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x34,0x01,0x88,0xc6,0xfa]

v_and_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x36,0x01,0x88,0xc6,0xfa]

v_or_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x38,0x01,0x88,0xc6,0xfa]

v_xor_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x3a,0x01,0x88,0xc6,0xfa]

v_xnor_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x3c,0x01,0x88,0xc6,0xfa]

v_add_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x64,0x01,0x88,0xc6,0xfa]

v_sub_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x66,0x01,0x88,0xc6,0xfa]

v_subrev_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x68,0x01,0x88,0xc6,0xfa]

v_mul_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x6a,0x01,0x88,0xc6,0xfa]

v_max_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x72,0x01,0x88,0xc6,0xfa]

v_min_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x74,0x01,0x88,0xc6,0xfa]

v_ldexp_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7]
// GFX10: encoding: [0xe9,0x04,0x0a,0x76,0x01,0x88,0xc6,0xfa]

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:0
// GFX10: encoding: [0xe9,0x02,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_mov_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x02,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_i32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x0a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_u32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x0c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_u32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x0e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x10,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x14,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x16,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_rpi_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x18,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_flr_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x1a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_off_f32_i4_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x1c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte0_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x22,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte1_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x24,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte2_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x26,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f32_ubyte3_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x28,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_fract_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x40,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_trunc_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x42,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ceil_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x44,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rndne_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x46,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_floor_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x48,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_exp_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x4a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_log_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x4e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x54,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_iflag_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x56,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rsq_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x5c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sqrt_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x66,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sin_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x6a,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cos_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x6c,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_not_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x6e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_bfrev_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x70,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbh_u32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x72,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbl_b32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x74,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ffbh_i32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x76,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_exp_i32_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x7e,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_mant_f32_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x80,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_u16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xa0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_f16_i16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xa2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_u16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xa4,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xa6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rcp_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xa8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sqrt_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xaa,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rsq_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xac,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_log_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xae,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_exp_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xb0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_mant_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xb2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_frexp_exp_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xb4,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_floor_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xb6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_ceil_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xb8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_trunc_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xba,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_rndne_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xbc,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_fract_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xbe,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_sin_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xc0,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cos_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xc2,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_norm_i16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xc6,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_cvt_norm_u16_f16_dpp v5, v1 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0xc8,0x0a,0x7e,0x01,0x88,0xc6,0xfa]

v_add_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x06,0x01,0x88,0xc6,0xfa]

v_sub_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x08,0x01,0x88,0xc6,0xfa]

v_subrev_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x0a,0x01,0x88,0xc6,0xfa]

v_mul_legacy_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x0e,0x01,0x88,0xc6,0xfa]

v_mul_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x10,0x01,0x88,0xc6,0xfa]

v_mul_i32_i24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x12,0x01,0x88,0xc6,0xfa]

v_mul_hi_i32_i24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x14,0x01,0x88,0xc6,0xfa]

v_mul_u32_u24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x16,0x01,0x88,0xc6,0xfa]

v_mul_hi_u32_u24_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x18,0x01,0x88,0xc6,0xfa]

v_min_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x1e,0x01,0x88,0xc6,0xfa]

v_max_f32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x20,0x01,0x88,0xc6,0xfa]

v_min_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x22,0x01,0x88,0xc6,0xfa]

v_max_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x24,0x01,0x88,0xc6,0xfa]

v_min_u32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x26,0x01,0x88,0xc6,0xfa]

v_max_u32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x28,0x01,0x88,0xc6,0xfa]

v_lshrrev_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x2c,0x01,0x88,0xc6,0xfa]

v_ashrrev_i32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x30,0x01,0x88,0xc6,0xfa]

v_lshlrev_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x34,0x01,0x88,0xc6,0xfa]

v_and_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x36,0x01,0x88,0xc6,0xfa]

v_or_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x38,0x01,0x88,0xc6,0xfa]

v_xor_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x3a,0x01,0x88,0xc6,0xfa]

v_xnor_b32_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x3c,0x01,0x88,0xc6,0xfa]

v_add_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x64,0x01,0x88,0xc6,0xfa]

v_sub_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x66,0x01,0x88,0xc6,0xfa]

v_subrev_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x68,0x01,0x88,0xc6,0xfa]

v_mul_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x6a,0x01,0x88,0xc6,0xfa]

v_max_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x72,0x01,0x88,0xc6,0xfa]

v_min_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x74,0x01,0x88,0xc6,0xfa]

v_ldexp_f16_dpp v5, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// GFX10: encoding: [0xea,0x04,0x0a,0x76,0x01,0x88,0xc6,0xfa]

v_cndmask_b32_dpp v0, v1, v2, vcc_lo dpp8:[7,6,5,4,3,2,1,0]
// W32: v_cndmask_b32_dpp v0, v1, v2, vcc_lo  dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x00,0x02,0x01,0x77,0x39,0x05]
// W64-ERR: error: operands are not valid for this GPU or mode

v_cndmask_b32_dpp v0, v1, v2, vcc_lo dpp8:[0,1,2,3,4,5,6,7] fi:1
// W32: v_cndmask_b32_dpp v0, v1, v2, vcc_lo  dpp8:[0,1,2,3,4,5,6,7] fi:1 ; encoding: [0xea,0x04,0x00,0x02,0x01,0x88,0xc6,0xfa]
// W64-ERR: error: operands are not valid for this GPU or mode

v_cndmask_b32_dpp v0, v1, v2, vcc dpp8:[7,6,5,4,3,2,1,0]
// W64: v_cndmask_b32_dpp v0, v1, v2, vcc  dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x00,0x02,0x01,0x77,0x39,0x05]
// W32-ERR: error: operands are not valid for this GPU or mode

v_cndmask_b32_dpp v0, v1, v2, vcc dpp8:[0,1,2,3,4,5,6,7] fi:1
// W64: v_cndmask_b32_dpp v0, v1, v2, vcc  dpp8:[0,1,2,3,4,5,6,7] fi:1 ; encoding: [0xea,0x04,0x00,0x02,0x01,0x88,0xc6,0xfa]
// W32-ERR: error: operands are not valid for this GPU or mode

v_cndmask_b32_dpp v0, v1, v2 dpp8:[0,1,2,3,4,5,6,7] fi:1
// W32: v_cndmask_b32_dpp v0, v1, v2, vcc_lo  dpp8:[0,1,2,3,4,5,6,7] fi:1 ; encoding: [0xea,0x04,0x00,0x02,0x01,0x88,0xc6,0xfa]
// W64: v_cndmask_b32_dpp v0, v1, v2, vcc  dpp8:[0,1,2,3,4,5,6,7] fi:1 ; encoding: [0xea,0x04,0x00,0x02,0x01,0x88,0xc6,0xfa]

v_add_co_ci_u32_dpp v0, vcc_lo, v0, v0, vcc_lo dpp8:[7,6,5,4,3,2,1,0]
// W32: [0xe9,0x00,0x00,0x50,0x00,0x77,0x39,0x05]
// W64-ERR: error: operands are not valid for this GPU or mode

v_sub_co_ci_u32_dpp v0, vcc_lo, v0, v0, vcc_lo dpp8:[7,6,5,4,3,2,1,0] fi:0
// W32: [0xe9,0x00,0x00,0x52,0x00,0x77,0x39,0x05]
// W64-ERR: error: operands are not valid for this GPU or mode

v_subrev_co_ci_u32_dpp v0, vcc_lo, v0, v0, vcc_lo dpp8:[7,6,5,4,3,2,1,0] fi:1
// W32: [0xea,0x00,0x00,0x54,0x00,0x77,0x39,0x05]
// W64-ERR: error: operands are not valid for this GPU or mode

v_add_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0] fi:1
// W64: [0xea,0x00,0x00,0x50,0x00,0x77,0x39,0x05]
// W32-ERR: error: operands are not valid for this GPU or mode

v_sub_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0] fi:1
// W64: [0xea,0x00,0x00,0x52,0x00,0x77,0x39,0x05]
// W32-ERR: error: operands are not valid for this GPU or mode

v_subrev_co_ci_u32_dpp v0, vcc, v0, v0, vcc dpp8:[7,6,5,4,3,2,1,0]
// W64: [0xe9,0x00,0x00,0x54,0x00,0x77,0x39,0x05]
// W32-ERR: error: operands are not valid for this GPU or mode

v_add_nc_u32_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: [0xe9,0xfe,0x0b,0x4a,0x01,0x77,0x39,0x05]

v_add_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: [0xea,0x04,0x0a,0x4a,0x01,0x77,0x39,0x05]

v_sub_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: [0xe9,0x04,0x0a,0x4c,0x01,0x77,0x39,0x05]

v_sub_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: [0xea,0x04,0x0a,0x4c,0x01,0x77,0x39,0x05]

v_subrev_nc_u32_dpp v5, v1, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: [0xe9,0xfe,0x0b,0x4e,0x01,0x77,0x39,0x05]

v_subrev_nc_u32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: [0xea,0x04,0x0a,0x4e,0x01,0x77,0x39,0x05]

v_mac_f32 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: v_mac_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] ; encoding: [0xe9,0x04,0x0a,0x3e,0x01,0x77,0x39,0x05]

v_mac_f32 v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: v_mac_f32_dpp v5, v1, v2 dpp8:[7,6,5,4,3,2,1,0] fi:1 ; encoding: [0xea,0x04,0x0a,0x3e,0x01,0x77,0x39,0x05]

v_movreld_b32 v0, v1 dpp8:[7,6,5,4,3,2,1,0] fi:1
// GFX10: [0xea,0x84,0x00,0x7e,0x01,0x77,0x39,0x05]

v_movrels_b32 v0, v2 dpp8:[0,0,0,0,0,0,0,0]
// GFX10: [0xe9,0x86,0x00,0x7e,0x02,0x00,0x00,0x00]

v_movrelsd_2_b32 v0, v255 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: [0xe9,0x90,0x00,0x7e,0xff,0x77,0x39,0x05]

v_movrelsd_b32 v0, v2 dpp8:[7,6,5,4,3,2,1,0]
// GFX10: [0xe9,0x88,0x00,0x7e,0x02,0x77,0x39,0x05]
