/*===-- OmpSs.h - OmpSs Transformation Library C Interface ----*- C++ -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMOmpSsOpts.a, which         *|
|* implements various scalar transformations of the LLVM IR.                  *|
|*                                                                            *|
|* Many exotic languages can interoperate with C code but have a harder time  *|
|* with C++ due to name mangling. So in addition to C, this interface enables *|
|* tools written in such languages.                                           *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_TRANSFORMS_OMPSS_H
#define LLVM_C_TRANSFORMS_OMPSS_H

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCTransformsOmpSs OmpSs transformations
 * @ingroup LLVMCTransforms
 *
 * @{
 */

/** See llvm::createOmpSsPass function. */
void LLVMOmpSsPass(LLVMPassManagerRef PM);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* defined(__cplusplus) */

#endif
