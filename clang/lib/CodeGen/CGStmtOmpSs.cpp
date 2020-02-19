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

static void AddDSASharedData(const OSSTaskDirective &S, SmallVectorImpl<const Expr *> &Data) {
  // All DSAs are DeclRefExpr or CXXThisExpr
  llvm::SmallSet<const ValueDecl *, 8> DeclExpr;
  for (const auto *C : S.getClausesOfKind<OSSSharedClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Ref)) {
        const ValueDecl *VD = DRE->getDecl();
        if (DeclExpr.insert(VD).second)
          Data.push_back(Ref);
      } else if (const CXXThisExpr *ThisE = dyn_cast<CXXThisExpr>(Ref)) {
        // 'this' expression is only allowed when it's generated by the compiler
        Data.push_back(ThisE);
      }
    }
  }
}

static void AddDSAPrivateData(const OSSTaskDirective &S, SmallVectorImpl<OSSDSAPrivateDataTy> &PList) {
  // All DSA are DeclRefExpr
  llvm::SmallSet<const ValueDecl *, 8> DeclExpr;
  for (const auto *C : S.getClausesOfKind<OSSPrivateClause>()) {
    auto CopyRef = C->private_copies().begin();
    for (const Expr *Ref : C->varlists()) {
      const DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Ref);
      const ValueDecl *VD = DRE->getDecl();
      // This assumes item i has a copyref and initref
      if (DeclExpr.insert(VD).second)
        PList.push_back({Ref, *CopyRef});

      ++CopyRef;
    }
  }
}

static void AddDSAFirstprivateData(const OSSTaskDirective &S,
                                   SmallVectorImpl<OSSDSAFirstprivateDataTy> &FpList) {
  // All DSA are DeclRefExpr
  llvm::SmallSet<const ValueDecl *, 8> DeclExpr;
  for (const auto *C : S.getClausesOfKind<OSSFirstprivateClause>()) {
    auto CopyRef = C->private_copies().begin();
    auto InitRef = C->inits().begin();
    for (const Expr *Ref : C->varlists()) {
      const DeclRefExpr *DRE = cast<DeclRefExpr>(Ref);
      const ValueDecl *VD = DRE->getDecl();
      // This assumes item i has a copyref and initref
      if (DeclExpr.insert(VD).second)
        FpList.push_back({Ref, *CopyRef, *InitRef});

      ++CopyRef;
      ++InitRef;
    }
  }
}

static void AddDSAData(const OSSTaskDirective &S, OSSTaskDSADataTy &DSAs) {
  AddDSASharedData(S, DSAs.Shareds);
  AddDSAPrivateData(S, DSAs.Privates);
  AddDSAFirstprivateData(S, DSAs.Firstprivates);
};

static void AddDepData(const OSSTaskDirective &S, OSSTaskDepDataTy &Deps) {
  for (const auto *C : S.getClausesOfKind<OSSDependClause>()) {
    ArrayRef<OmpSsDependClauseKind> DepKindsOrdered = C->getDependencyKindsOrdered();
    if (DepKindsOrdered.size() == 2) {
      for (const Expr *Ref : C->varlists()) {
        if (DepKindsOrdered[0] == OSSC_DEPEND_in)
          Deps.WeakIns.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_out)
          Deps.WeakOuts.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_inout)
          Deps.WeakInouts.push_back({C->isOSSSyntax(), Ref});
      }
    }
    else {
      for (const Expr *Ref : C->varlists()) {
        if (DepKindsOrdered[0] == OSSC_DEPEND_in)
          Deps.Ins.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_out)
          Deps.Outs.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_inout)
          Deps.Inouts.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_mutexinoutset)
          Deps.Concurrents.push_back({C->isOSSSyntax(), Ref});
        if (DepKindsOrdered[0] == OSSC_DEPEND_inoutset)
          Deps.Commutatives.push_back({C->isOSSSyntax(), Ref});
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

static void AddCostData(const OSSTaskDirective &S, const Expr * &CostExpr) {
  bool Found = false;
  for (const auto *C : S.getClausesOfKind<OSSCostClause>()) {
    assert(!Found);
    Found = true;
    CostExpr = C->getExpression();
  }
}

static void AddPriorityData(const OSSTaskDirective &S, const Expr * &PriorityExpr) {
  bool Found = false;
  for (const auto *C : S.getClausesOfKind<OSSPriorityClause>()) {
    assert(!Found);
    Found = true;
    PriorityExpr = C->getExpression();
  }
}

static void AddReductionData(const OSSTaskDirective &S, OSSTaskReductionDataTy &Reductions) {
  for (const auto *C : S.getClausesOfKind<OSSReductionClause>()) {
    auto SimpleRef = C->simple_exprs().begin();
    auto InitRef = C->privates().begin();
    auto LHSRef = C->lhs_exprs().begin();
    auto RHSRef = C->rhs_exprs().begin();
    auto RedOp = C->reduction_ops().begin();
    for (const Expr *Ref : C->varlists()) {
      Reductions.RedList.push_back({*SimpleRef, Ref, *InitRef, *LHSRef, *RHSRef, *RedOp});

      ++SimpleRef;
      ++InitRef;
      ++LHSRef;
      ++RHSRef;
      ++RedOp;
    }
  }
}

void CodeGenFunction::EmitOSSTaskDirective(const OSSTaskDirective &S) {
  OSSTaskDataTy Data;

  AddDSAData(S, Data.DSAs);
  AddDepData(S, Data.Deps);
  AddIfData(S, Data.If);
  AddFinalData(S, Data.Final);
  AddCostData(S, Data.Cost);
  AddPriorityData(S, Data.Priority);
  AddReductionData(S, Data.Reductions);

  CGM.getOmpSsRuntime().emitTaskCall(*this, S, S.getBeginLoc(), Data);
}

