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

template<typename DSAKind>
static void AddDSAData(const OSSTaskDirective &S, SmallVectorImpl<const Expr *> &Data) {
  // All DSA are DeclRefExpr
  llvm::SmallSet<const ValueDecl *, 8> DeclExpr;
  for (const auto *C : S.getClausesOfKind<DSAKind>()) {
      for (const Expr *Ref : C->varlists()) {
        if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Ref)) {
          const ValueDecl *VD = DRE->getDecl();
          if (DeclExpr.insert(VD).second)
            Data.push_back(Ref);
        }
      }
  }
}

static void AddDSAData(const OSSTaskDirective &S, OSSDSADataTy &DSAs) {
  AddDSAData<OSSSharedClause>(S, DSAs.Shareds);
  AddDSAData<OSSPrivateClause>(S, DSAs.Privates);
  AddDSAData<OSSFirstprivateClause>(S, DSAs.Firstprivates);
};

static void AddDepData(const OSSTaskDirective &S, OSSDepDataTy &Deps) {
  for (const auto *C : S.getClausesOfKind<OSSDependClause>()) {
    ArrayRef<OmpSsDependClauseKind> DepKinds = C->getDependencyKind();
    if (DepKinds.size() == 2) {
      for (const Expr *Ref : C->varlists()) {
        if (DepKinds[0] == OSSC_DEPEND_in
            || DepKinds[1] == OSSC_DEPEND_in)
          Deps.WeakIns.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_out
            || DepKinds[1] == OSSC_DEPEND_out)
          Deps.WeakOuts.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_inout
            || DepKinds[1] == OSSC_DEPEND_inout)
          Deps.WeakInouts.push_back(Ref);
      }
    }
    else {
      for (const Expr *Ref : C->varlists()) {
        if (DepKinds[0] == OSSC_DEPEND_in)
          Deps.Ins.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_out)
          Deps.Outs.push_back(Ref);
        if (DepKinds[0] == OSSC_DEPEND_inout)
          Deps.Inouts.push_back(Ref);
      }
    }
  }
};

static void AddIfData(const OSSTaskDirective &S, const Expr *&IfExpr) {
  bool Found = false;
  for (const auto *C : S.getClausesOfKind<OSSIfClause>()) {
    assert(!Found);
    Found = true;
    IfExpr = C->getCondition();
  }
}

static void AddFinalData(const OSSTaskDirective &S, const Expr * &FinalExpr) {
  bool Found = false;
  for (const auto *C : S.getClausesOfKind<OSSFinalClause>()) {
    assert(!Found);
    Found = true;
    FinalExpr = C->getCondition();
  }
}

void CodeGenFunction::EmitOSSTaskDirective(const OSSTaskDirective &S) {
  OSSTaskDataTy Data;

  AddDSAData(S, Data.DSAs);
  AddDepData(S, Data.Deps);
  AddIfData(S, Data.If);
  AddFinalData(S, Data.Final);

  CGM.getOmpSsRuntime().emitTaskCall(*this, S, S.getBeginLoc(), Data);
}

