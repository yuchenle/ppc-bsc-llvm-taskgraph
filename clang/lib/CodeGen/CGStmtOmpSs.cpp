//===--- CGStmtOmpSs.cpp - Emit LLVM Code from Statements ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit OmpSs nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGCleanup.h"
#include "CGOmpSsRuntime.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOmpSs.h"
#include "llvm/IR/CallSite.h"
using namespace clang;
using namespace CodeGen;

void CodeGenFunction::EmitOSSTaskwaitDirective(const OSSTaskwaitDirective &S) {
  CGM.getOmpSsRuntime().emitTaskwaitCall(*this, S.getBeginLoc());
}

void CodeGenFunction::EmitOSSTaskDirective(const OSSTaskDirective &S) {
  OSSTaskDataTy Data;
  for (const auto *C : S.getClausesOfKind<OSSSharedClause>()) {
      for (const Expr *Ref : C->varlists())
          Data.SharedVars.push_back(Ref);
  }
  for (const auto *C : S.getClausesOfKind<OSSPrivateClause>()) {
      for (const Expr *Ref : C->varlists())
          Data.PrivateVars.push_back(Ref);
  }
  for (const auto *C : S.getClausesOfKind<OSSFirstprivateClause>()) {
      for (const Expr *Ref : C->varlists())
          Data.FirstprivateVars.push_back(Ref);
  }

  CGM.getOmpSsRuntime().emitTaskCall(*this, S, S.getBeginLoc(), Data);
}

