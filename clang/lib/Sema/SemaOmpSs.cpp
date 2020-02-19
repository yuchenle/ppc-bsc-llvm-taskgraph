//===--- SemaOmpSs.cpp - Semantic Analysis for OmpSs constructs ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OmpSs directives and
/// clauses.
///
//===----------------------------------------------------------------------===//

#include "TreeTransform.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/StmtOmpSs.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/SemaInternal.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
using namespace clang;

namespace {
/// Default data sharing attributes, which can be applied to directive.
enum DefaultDataSharingAttributes {
  DSA_unspecified = 0, /// Data sharing attribute not specified.
  DSA_none = 1 << 0,   /// Default data sharing attribute 'none'.
  DSA_shared = 1 << 1, /// Default data sharing attribute 'shared'.
};

/// Stack for tracking declarations used in OmpSs directives and
/// clauses and their data-sharing attributes.
class DSAStackTy {
public:
  struct DSAVarData {
    OmpSsDirectiveKind DKind = OSSD_unknown;
    OmpSsClauseKind CKind = OSSC_unknown;
    const Expr *RefExpr = nullptr;
    bool Ignore = false;
    bool Implicit = false;
    DSAVarData() = default;
    DSAVarData(OmpSsDirectiveKind DKind, OmpSsClauseKind CKind,
               const Expr *RefExpr, bool Ignore)
        : DKind(DKind), CKind(CKind), RefExpr(RefExpr), Ignore(Ignore)
          {}
  };

private:
  struct DSAInfo {
    OmpSsClauseKind Attributes = OSSC_unknown;
    const Expr * RefExpr;
    bool Ignore = false;
    bool Implicit = false;
  };
  using DeclSAMapTy = llvm::SmallDenseMap<const ValueDecl *, DSAInfo, 8>;

  // Directive
  struct SharingMapTy {
    DeclSAMapTy SharingMap;
    DefaultDataSharingAttributes DefaultAttr = DSA_unspecified;
    SourceLocation DefaultAttrLoc;
    OmpSsDirectiveKind Directive = OSSD_unknown;
    Scope *CurScope = nullptr;
    CXXThisExpr *ThisExpr = nullptr;
    SourceLocation ConstructLoc;
    SharingMapTy(OmpSsDirectiveKind DKind,
                 Scope *CurScope, SourceLocation Loc)
        : Directive(DKind), CurScope(CurScope),
          ConstructLoc(Loc) {}
    SharingMapTy() = default;
  };

  using StackTy = SmallVector<SharingMapTy, 4>;

  /// Stack of used declaration and their data-sharing attributes.
  StackTy Stack;
  Sema &SemaRef;

  using iterator = StackTy::const_reverse_iterator;

  DSAVarData getDSA(iterator &Iter, ValueDecl *D) const;

  bool isStackEmpty() const {
    return Stack.empty();
  }

public:
  explicit DSAStackTy(Sema &S) : SemaRef(S) {}

  void push(OmpSsDirectiveKind DKind,
            Scope *CurScope, SourceLocation Loc) {
    Stack.emplace_back(DKind, CurScope, Loc);
  }

  void pop() {
    assert(!Stack.empty() && "Data-sharing attributes stack is empty!");
    Stack.pop_back();
  }

  /// Adds explicit data sharing attribute to the specified declaration.
  void addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A,
              bool Ignore, bool Implicit);

  /// Returns data sharing attributes from top of the stack for the
  /// specified declaration.
  const DSAVarData getTopDSA(ValueDecl *D, bool FromParent);
  /// Returns data sharing attributes from the current directive for the
  /// specified declaration.
  const DSAVarData getCurrentDSA(ValueDecl *D);
  /// Returns data-sharing attributes for the specified declaration.
  /// Checks if the specified variables has data-sharing attributes which
  /// match specified \a CPred predicate in any directive which matches \a DPred
  /// predicate.
  const DSAVarData
  hasDSA(ValueDecl *D, const llvm::function_ref<bool(OmpSsClauseKind)> CPred,
         const llvm::function_ref<bool(OmpSsDirectiveKind)> DPred,
         bool FromParent) const;
  /// Set default data sharing attribute to none.
  void setDefaultDSANone(SourceLocation Loc) {
    assert(!isStackEmpty());
    Stack.back().DefaultAttr = DSA_none;
    Stack.back().DefaultAttrLoc = Loc;
  }
  /// Set default data sharing attribute to shared.
  void setDefaultDSAShared(SourceLocation Loc) {
    assert(!isStackEmpty());
    Stack.back().DefaultAttr = DSA_shared;
    Stack.back().DefaultAttrLoc = Loc;
  }
  void setThisExpr(CXXThisExpr *ThisE) {
    Stack.back().ThisExpr = ThisE;
  }
  /// Returns currently analyzed directive.
  OmpSsDirectiveKind getCurrentDirective() const {
    return isStackEmpty() ? OSSD_unknown : Stack.back().Directive;
  }
  DefaultDataSharingAttributes getCurrentDefaultDataSharingAttributtes() const {
    return isStackEmpty() ? DSA_unspecified : Stack.back().DefaultAttr;
  }
  CXXThisExpr *getThisExpr() const {
    return isStackEmpty() ? nullptr : Stack.back().ThisExpr;
  }
};

} // namespace

static const ValueDecl *getCanonicalDecl(const ValueDecl *D) {
  const auto *VD = dyn_cast<VarDecl>(D);
  const auto *FD = dyn_cast<FieldDecl>(D);
  if (VD != nullptr) {
    VD = VD->getCanonicalDecl();
    D = VD;
  } else {
    assert(FD);
    FD = FD->getCanonicalDecl();
    D = FD;
  }
  return D;
}

static ValueDecl *getCanonicalDecl(ValueDecl *D) {
  return const_cast<ValueDecl *>(
      getCanonicalDecl(const_cast<const ValueDecl *>(D)));
}

DSAStackTy::DSAVarData DSAStackTy::getDSA(iterator &Iter,
                                          ValueDecl *D) const {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  DVar.DKind = Iter->Directive;
  if (Iter->SharingMap.count(D)) {
    const DSAInfo &Data = Iter->SharingMap.lookup(D);
    DVar.RefExpr = Data.RefExpr;
    DVar.Ignore = Data.Ignore;
    DVar.Implicit = Data.Implicit;
    DVar.CKind = Data.Attributes;
    return DVar;
  }

  return DVar;
}

void DSAStackTy::addDSA(const ValueDecl *D, const Expr *E, OmpSsClauseKind A,
                        bool Ignore, bool Implicit) {
  D = getCanonicalDecl(D);
  assert(!isStackEmpty() && "Data-sharing attributes stack is empty");
  DSAInfo &Data = Stack.back().SharingMap[D];
  Data.Attributes = A;
  Data.RefExpr = E;
  Data.Ignore = Ignore;
  Data.Implicit = Implicit;
}

const DSAStackTy::DSAVarData DSAStackTy::getTopDSA(ValueDecl *D,
                                                   bool FromParent) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return Dir == OSSD_task; };
  auto &&AnyClause = [](OmpSsClauseKind Clause) { return Clause != OSSC_shared; };
  if (VD) {
    DSAVarData DVarTemp = hasDSA(D, AnyClause, IsTaskDir, FromParent);
    if (DVarTemp.CKind != OSSC_unknown && DVarTemp.RefExpr)
      return DVarTemp;
  }

  return DVar;
}

const DSAStackTy::DSAVarData DSAStackTy::getCurrentDSA(ValueDecl *D) {
  D = getCanonicalDecl(D);
  DSAVarData DVar;

  auto *VD = dyn_cast<VarDecl>(D);

  auto &&IsTaskDir = [](OmpSsDirectiveKind Dir) { return Dir == OSSD_task; };
  auto &&AnyClause = [](OmpSsClauseKind Clause) { return true; };
  iterator I = Stack.rbegin();
  iterator EndI = Stack.rend();
  if (VD){
    if (I != EndI) {
      if (IsTaskDir(I->Directive)) {
        DSAVarData DVar = getDSA(I, D);
          if (AnyClause(DVar.CKind))
            return DVar;
      }
    }
  }
  return DVar;
}

const DSAStackTy::DSAVarData
DSAStackTy::hasDSA(ValueDecl *D,
                   const llvm::function_ref<bool(OmpSsClauseKind)> CPred,
                   const llvm::function_ref<bool(OmpSsDirectiveKind)> DPred,
                   bool FromParent) const {
  D = getCanonicalDecl(D);
  iterator I = Stack.rbegin();
  iterator EndI = Stack.rend();
  if (FromParent && I != EndI)
    std::advance(I, 1);
  for (; I != EndI; std::advance(I, 1)) {
    if (!DPred(I->Directive))
      continue;
    DSAVarData DVar = getDSA(I, D);
    if (CPred(DVar.CKind))
      return DVar;
  }
  return {};
}

namespace {
class DSAAttrChecker final : public StmtVisitor<DSAAttrChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound = false;
  Stmt *CS = nullptr;
  llvm::SmallVector<Expr *, 4> ImplicitShared;
  llvm::SmallVector<Expr *, 4> ImplicitFirstprivate;
  llvm::SmallSet<ValueDecl *, 4> InnerDecls;

  // Walks over all array dimensions looking for VLA size Expr.
  void GetTypeDSAs(QualType T) {
    QualType TmpTy = T;
    // int (**p)[sizex][sizey] -> we need sizex sizey for vla dims
    while (TmpTy->isPointerType())
      TmpTy = TmpTy->getPointeeType();
    while (TmpTy->isArrayType()) {
      if (const ConstantArrayType *BaseArrayTy = SemaRef.Context.getAsConstantArrayType(TmpTy)) {
        TmpTy = BaseArrayTy->getElementType();
      } else if (const VariableArrayType *BaseArrayTy = SemaRef.Context.getAsVariableArrayType(TmpTy)) {
        Expr *SizeExpr = BaseArrayTy->getSizeExpr();
        Visit(SizeExpr);
        TmpTy = BaseArrayTy->getElementType();
      } else {
        llvm_unreachable("Unhandled array type");
      }
    }
  }

public:

  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // Add DSA to 'this' if is the first time we see it
    if (!Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
      ImplicitShared.push_back(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;
    if (E->isNonOdrUse())
      return;
    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();

      // Variables declared inside region don't have DSA
      if (InnerDecls.count(VD))
        return;

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      DSAStackTy::DSAVarData DVarFromParent = Stack->getTopDSA(VD, /*FromParent=*/true);

      bool ExistsParent = DVarFromParent.RefExpr;
      bool ParentIgnore = DVarFromParent.Ignore;

      bool ExistsCurrent = DVarCurrent.RefExpr;

      // Check if the variable has DSA set on the current
      // directive and stop analysis if it so.
      if (ExistsCurrent)
        return;
      // If explicit DSA comes from parent inherit it
      if (ExistsParent && !ParentIgnore) {
          switch (DVarFromParent.CKind) {
          case OSSC_shared:
            ImplicitShared.push_back(E);
            break;
          case OSSC_private:
          case OSSC_firstprivate:
            ImplicitFirstprivate.push_back(E);
            break;
          default:
            llvm_unreachable("unexpected DSA from parent");
          }
      } else {

        // QualType Type = VD->getType();
        // if (!(Type.isPODType(SemaRef.Context) || Type->isReferenceType())) {
        //   return;
        // }

        OmpSsDirectiveKind DKind = Stack->getCurrentDirective();
        switch (Stack->getCurrentDefaultDataSharingAttributtes()) {
        case DSA_shared:
          // Define implicit data-sharing attributes for task.
          if (isOmpSsTaskingDirective(DKind))
            ImplicitShared.push_back(E);
          // Record DSA as Ignored to avoid making the same node again
          Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/true, /*Implicit=*/true);
          break;
        case DSA_none:
          if (!DVarCurrent.Ignore) {
            SemaRef.Diag(E->getExprLoc(), diag::err_oss_not_defined_dsa_when_default_none) << E->getDecl();
            // Record DSA as ignored to diagnostic only once
            Stack->addDSA(VD, E, OSSC_unknown, /*Ignore=*/true, /*Implicit=*/true);
          }
          break;
        case DSA_unspecified:
          if (VD->hasLocalStorage()) {
            // If no default clause is present and the variable was private/local
            // in the context encountering the construct, the variable will
            // be firstprivate

            // Define implicit data-sharing attributes for task.
            if (isOmpSsTaskingDirective(DKind))
              ImplicitFirstprivate.push_back(E);

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_firstprivate, /*Ignore=*/true, /*Implicit=*/true);
          } else {
            // If no default clause is present and the variable was shared/global
            // in the context encountering the construct, the variable will be shared.

            // Define implicit data-sharing attributes for task.
            if (isOmpSsTaskingDirective(DKind))
              ImplicitShared.push_back(E);

            // Record DSA as Ignored to avoid making the same node again
            Stack->addDSA(VD, E, OSSC_shared, /*Ignore=*/true, /*Implicit=*/true);
          }
        }
      }
    }
  }

  void VisitCXXCatchStmt(CXXCatchStmt *Node) {
    InnerDecls.insert(Node->getExceptionDecl());
    Visit(Node->getHandlerBlock());
  }

  void VisitExpr(Expr *E) {
    for (Stmt *Child : E->children()) {
      if (Child)
        Visit(Child);
    }
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  void VisitDeclStmt(DeclStmt *S) {
    for (Decl *D : S->decls()) {
      if (auto *VD = dyn_cast_or_null<VarDecl>(D)) {
        InnerDecls.insert(VD);
        if (VD->hasInit()) {
          Visit(VD->getInit());
        }
        GetTypeDSAs(VD->getType());
      }
    }
  }

  bool isErrorFound() const { return ErrorFound; }

  ArrayRef<Expr *> getImplicitShared() const {
    return ImplicitShared;
  }

  ArrayRef<Expr *> getImplicitFirstprivate() const {
    return ImplicitFirstprivate;
  }

  DSAAttrChecker(DSAStackTy *S, Sema &SemaRef, Stmt *CS)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false), CS(CS) {}
};

// OSSClauseDSAChecker gathers for each expression in a clause
// all implicit data-sharings.
//
// To do so, we classify as firstprivate the base symbol if it's a pointer and is
// dereferenced by a SubscriptExpr, MemberExpr or UnaryOperator.
// Otherwise it's shared.
//
// At the same time, all symbols found inside a SubscriptExpr will be firstprivate.
// NOTE: implicit DSA from other tasks are ignored
class OSSClauseDSAChecker final : public StmtVisitor<OSSClauseDSAChecker, void> {
  DSAStackTy *Stack;
  Sema &SemaRef;
  bool ErrorFound = false;
  llvm::SmallVector<Expr *, 4> ImplicitFirstprivate;
  llvm::SmallVector<Expr *, 4> ImplicitShared;
  // This is used to know we're inside a subscript expression
  size_t ArraySubscriptCnt = 0;
  // This is used to mark the innermost base symbol expression as:
  // *p, p[2], p[1:2], [2]p, s.x, s->x
  bool IsDerefMemberArrayBase = false;
public:

  void VisitOSSArrayShapingExpr(OSSArrayShapingExpr *E) {
    if (isa<DeclRefExpr>(E->getBase()->IgnoreParenImpCasts()))
      IsDerefMemberArrayBase = true;
    Visit(E->getBase());
    IsDerefMemberArrayBase = false;

    ArraySubscriptCnt++;
    for (Stmt *S : E->getShapes())
      Visit(S);
    ArraySubscriptCnt--;
  }

  void VisitOSSArraySectionExpr(OSSArraySectionExpr *E) {
    if (isa<DeclRefExpr>(E->getBase()->IgnoreParenImpCasts()))
      IsDerefMemberArrayBase = true;
    Visit(E->getBase());
    IsDerefMemberArrayBase = false;

    ArraySubscriptCnt++;
    if (E->getLowerBound())
      Visit(E->getLowerBound());
    if (E->getLengthUpper())
      Visit(E->getLengthUpper());
    ArraySubscriptCnt--;
  }

  void VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
    if (isa<DeclRefExpr>(E->getBase()->IgnoreParenImpCasts()))
      IsDerefMemberArrayBase = true;
    Visit(E->getBase());
    IsDerefMemberArrayBase = false;

    ArraySubscriptCnt++;
    Visit(E->getIdx());
    ArraySubscriptCnt--;
  }

  void VisitUnaryOperator(UnaryOperator *E) {
    if (isa<DeclRefExpr>(E->getSubExpr()->IgnoreParenImpCasts()))
      IsDerefMemberArrayBase = true;
    Visit(E->getSubExpr());
    IsDerefMemberArrayBase = false;
  }

  void VisitMemberExpr(MemberExpr *E) {
    if (isa<DeclRefExpr>(E->getBase()->IgnoreParenImpCasts()))
      IsDerefMemberArrayBase = true;
    Visit(E->getBase());
    IsDerefMemberArrayBase = false;
  }

  void VisitCXXThisExpr(CXXThisExpr *ThisE) {
    // Add DSA to 'this' if is the first time we see it
    if (!Stack->getThisExpr()) {
      Stack->setThisExpr(ThisE);
      ImplicitShared.push_back(ThisE);
    }
  }
  void VisitDeclRefExpr(DeclRefExpr *E) {
    if (E->isTypeDependent() || E->isValueDependent() ||
        E->containsUnexpandedParameterPack() || E->isInstantiationDependent())
      return;

    if (auto *VD = dyn_cast<VarDecl>(E->getDecl())) {
      VD = VD->getCanonicalDecl();
      // inout(x)              | shared(x)        | int x;
      // inout(p[i])           | firstprivate(p)  | int *p;
      // inout(a[i])           | shared(a)        | int a[N];
      // inout(*p)/inout(p[0]) | firstprivate(p)  | int *p;
      // inout(s.x)            | shared(s)        | struct S s;
      // inout(ps->x)          | firstprivate(ps) | struct S *ps;
      // inout([1]p)           | firstprivate(p)  | int *p;
      OmpSsClauseKind VKind = OSSC_shared;
      // FIXME?: There's an overlapping between IsDerefMemberArrayBase
      // and ArraySubscriptCnt
      // i.e
      //    a[b[7]]
      // b will have ArraySubscriptCnt > 0
      // and IsDerefMemberArrayBase true
      // Check ArraySubscriptCnt first since is more restrictive
      if (ArraySubscriptCnt)
        VKind = OSSC_firstprivate;
      else if (VD->getType()->isPointerType() && IsDerefMemberArrayBase)
        VKind = OSSC_firstprivate;

      SourceLocation ELoc = E->getExprLoc();
      SourceRange ERange = E->getSourceRange();

      DSAStackTy::DSAVarData DVarCurrent = Stack->getCurrentDSA(VD);
      switch (DVarCurrent.CKind) {
      case OSSC_shared:
        // Do nothing
        break;
      case OSSC_private:
        break;
      case OSSC_firstprivate:
        if (VKind == OSSC_shared) {
          if (DVarCurrent.Implicit) {
            // Promote Implicit firstprivate to Implicit shared
            auto It = ImplicitFirstprivate.begin();
            while (It != ImplicitFirstprivate.end()) {
              if (*It == DVarCurrent.RefExpr) break;
              ++It;
            }
            assert(It != ImplicitFirstprivate.end());
            ImplicitFirstprivate.erase(It);

            ImplicitShared.push_back(E);
            // Rewrite DSA
            Stack->addDSA(VD, E, VKind, /*Ignore=*/false, /*Implicit=*/true);
          } else {
            ErrorFound = true;
            SemaRef.Diag(ELoc, diag::err_oss_mismatch_depend_dsa)
              << getOmpSsClauseName(DVarCurrent.CKind)
              << getOmpSsClauseName(VKind) << ERange;
          }
        }
        break;
      case OSSC_unknown:
        if (VKind == OSSC_shared)
          ImplicitShared.push_back(E);
        if (VKind == OSSC_firstprivate)
          ImplicitFirstprivate.push_back(E);

        Stack->addDSA(VD, E, VKind, /*Ignore=*/false, /*Implicit=*/true);

        break;
      default:
        llvm_unreachable("unexpected DSA");
      }
    }
  }

  void VisitClause(OSSClause *Clause) {
    for (Stmt *Child : Clause->children()) {
      if (Child)
        Visit(Child);
    }
  }

  void VisitStmt(Stmt *S) {
    for (Stmt *C : S->children()) {
      if (C)
        Visit(C);
    }
  }

  bool isErrorFound() const { return ErrorFound; }

  OSSClauseDSAChecker(DSAStackTy *S, Sema &SemaRef)
      : Stack(S), SemaRef(SemaRef), ErrorFound(false) {}

  ArrayRef<Expr *> getImplicitShared() const {
    return ImplicitShared;
  }

  ArrayRef<Expr *> getImplicitFirstprivate() const {
    return ImplicitFirstprivate;
  }
};
} // namespace

static VarDecl *buildVarDecl(Sema &SemaRef, SourceLocation Loc, QualType Type,
                             StringRef Name, const AttrVec *Attrs = nullptr) {
  DeclContext *DC = SemaRef.CurContext;
  IdentifierInfo *II = &SemaRef.PP.getIdentifierTable().get(Name);
  TypeSourceInfo *TInfo = SemaRef.Context.getTrivialTypeSourceInfo(Type, Loc);
  auto *Decl =
      VarDecl::Create(SemaRef.Context, DC, Loc, Loc, II, Type, TInfo, SC_None);
  if (Attrs) {
    for (specific_attr_iterator<AlignedAttr> I(Attrs->begin()), E(Attrs->end());
         I != E; ++I)
      Decl->addAttr(*I);
  }
  Decl->setImplicit();
  return Decl;
}

static DeclRefExpr *buildDeclRefExpr(Sema &S, VarDecl *D, QualType Ty,
                                     SourceLocation Loc,
                                     bool RefersToCapture = false) {
  D->setReferenced();
  D->markUsed(S.Context);
  return DeclRefExpr::Create(S.getASTContext(), NestedNameSpecifierLoc(),
                             SourceLocation(), D, RefersToCapture, Loc, Ty,
                             VK_LValue);
}

void Sema::InitDataSharingAttributesStackOmpSs() {
  VarDataSharingAttributesStackOmpSs = new DSAStackTy(*this);
  // TODO: use another function
  AllowShapings = false;
}

#define DSAStack static_cast<DSAStackTy *>(VarDataSharingAttributesStackOmpSs)

void Sema::DestroyDataSharingAttributesStackOmpSs() { delete DSAStack; }

void Sema::StartOmpSsDSABlock(OmpSsDirectiveKind DKind,
                              Scope *CurScope, SourceLocation Loc) {
  DSAStack->push(DKind, CurScope, Loc);
  PushExpressionEvaluationContext(
      ExpressionEvaluationContext::PotentiallyEvaluated);
}

void Sema::EndOmpSsDSABlock(Stmt *CurDirective) {
  DSAStack->pop();
  DiscardCleanupsInEvaluationContext();
  PopExpressionEvaluationContext();
}

static std::string
getListOfPossibleValues(OmpSsClauseKind K, unsigned First, unsigned Last,
                        ArrayRef<unsigned> Exclude = llvm::None) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  unsigned Bound = Last >= 2 ? Last - 2 : 0;
  unsigned Skipped = Exclude.size();
  auto S = Exclude.begin(), E = Exclude.end();
  for (unsigned I = First; I < Last; ++I) {
    if (std::find(S, E, I) != E) {
      --Skipped;
      continue;
    }
    Out << "'" << getOmpSsSimpleClauseTypeName(K, I) << "'";
    if (I == Bound - Skipped)
      Out << " or ";
    else if (I != Bound + 1 - Skipped)
      Out << ", ";
  }
  return Out.str();
}

void Sema::ActOnOmpSsAfterClauseGathering(SmallVectorImpl<OSSClause *>& Clauses) {

  bool ErrorFound = false;

  OSSClauseDSAChecker OSSClauseChecker(DSAStack, *this);
  for (auto *Clause : Clauses) {
    if (isa<OSSDependClause>(Clause) || isa<OSSReductionClause>(Clause)) {
      OSSClauseChecker.VisitClause(Clause);
    }
    // FIXME: how to handle an error?
    if (OSSClauseChecker.isErrorFound())
      ErrorFound = true;
  }

  SmallVector<Expr *, 4> ImplicitShared(
      OSSClauseChecker.getImplicitShared().begin(),
      OSSClauseChecker.getImplicitShared().end());

  SmallVector<Expr *, 4> ImplicitFirstprivate(
      OSSClauseChecker.getImplicitFirstprivate().begin(),
      OSSClauseChecker.getImplicitFirstprivate().end());

  if (!ImplicitShared.empty()) {
    if (OSSClause *Implicit = ActOnOmpSsSharedClause(
            ImplicitShared, SourceLocation(), SourceLocation(),
            SourceLocation(), /*isImplicit=*/true)) {
      Clauses.push_back(Implicit);
      if (cast<OSSSharedClause>(Implicit)->varlist_size() != ImplicitShared.size())
        ErrorFound = true;

    } else {
      ErrorFound = true;
    }
  }

  if (!ImplicitFirstprivate.empty()) {
    if (OSSClause *Implicit = ActOnOmpSsFirstprivateClause(
            ImplicitFirstprivate, SourceLocation(), SourceLocation(),
            SourceLocation())) {
      Clauses.push_back(Implicit);
      if (cast<OSSFirstprivateClause>(Implicit)->varlist_size() != ImplicitFirstprivate.size())
        ErrorFound = true;
    } else {
      ErrorFound = true;
    }
  }
}

StmtResult Sema::ActOnOmpSsExecutableDirective(ArrayRef<OSSClause *> Clauses,
    OmpSsDirectiveKind Kind, Stmt *AStmt, SourceLocation StartLoc, SourceLocation EndLoc) {

  bool ErrorFound = false;

  llvm::SmallVector<OSSClause *, 8> ClausesWithImplicit;
  ClausesWithImplicit.append(Clauses.begin(), Clauses.end());
  if (AStmt && !CurContext->isDependentContext()) {
    // Check default data sharing attributes for referenced variables.
    DSAAttrChecker DSAChecker(DSAStack, *this, AStmt);
    Stmt *S = AStmt;
    DSAChecker.Visit(S);
    if (DSAChecker.isErrorFound())
      ErrorFound = true;

    SmallVector<Expr *, 4> ImplicitShared(
        DSAChecker.getImplicitShared().begin(),
        DSAChecker.getImplicitShared().end());

    SmallVector<Expr *, 4> ImplicitFirstprivate(
        DSAChecker.getImplicitFirstprivate().begin(),
        DSAChecker.getImplicitFirstprivate().end());

    if (!ImplicitShared.empty()) {
      if (OSSClause *Implicit = ActOnOmpSsSharedClause(
              ImplicitShared, SourceLocation(), SourceLocation(),
              SourceLocation(), /*isImplicit=*/true)) {
        ClausesWithImplicit.push_back(Implicit);
        if (cast<OSSSharedClause>(Implicit)->varlist_size() != ImplicitShared.size())
          ErrorFound = true;
      } else {
        ErrorFound = true;
      }
    }

    if (!ImplicitFirstprivate.empty()) {
      if (OSSClause *Implicit = ActOnOmpSsFirstprivateClause(
              ImplicitFirstprivate, SourceLocation(), SourceLocation(),
              SourceLocation())) {
        ClausesWithImplicit.push_back(Implicit);
        if (cast<OSSFirstprivateClause>(Implicit)->varlist_size() != ImplicitFirstprivate.size())
          ErrorFound = true;
      } else {
        ErrorFound = true;
      }
    }
  }

  StmtResult Res = StmtError();
  switch (Kind) {
  case OSSD_taskwait:
    Res = ActOnOmpSsTaskwaitDirective(StartLoc, EndLoc);
    break;
  case OSSD_task:
    Res = ActOnOmpSsTaskDirective(ClausesWithImplicit, AStmt, StartLoc, EndLoc);
    break;
  case OSSD_unknown:
    llvm_unreachable("Unknown OmpSs directive");
  }

  if (ErrorFound)
    return StmtError();

  return Res;
}

StmtResult Sema::ActOnOmpSsTaskwaitDirective(SourceLocation StartLoc,
                                             SourceLocation EndLoc) {
  return OSSTaskwaitDirective::Create(Context, StartLoc, EndLoc);
}

StmtResult Sema::ActOnOmpSsTaskDirective(ArrayRef<OSSClause *> Clauses,
                                         Stmt *AStmt,
                                         SourceLocation StartLoc,
                                         SourceLocation EndLoc) {
  if (!AStmt)
    return StmtError();
  return OSSTaskDirective::Create(Context, StartLoc, EndLoc, Clauses, AStmt);
}

static void checkOutlineDependency(Sema &S, Expr *RefExpr, bool OSSSyntax=false) {
  SourceLocation ELoc = RefExpr->getExprLoc();
  Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
  if (RefExpr->isTypeDependent() || RefExpr->isValueDependent() ||
      RefExpr->containsUnexpandedParameterPack()) {
    // It will be analyzed later.
    return;
  }
  auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
  if (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
      (ASE &&
       !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
       !ASE->getBase()->getType().getNonReferenceType()->isArrayType())) {
    S.Diag(ELoc, diag::err_oss_expected_dereference_or_array_item)
        << RefExpr->getSourceRange();
    return;
  }
  if (isa<DeclRefExpr>(SimpleExpr) || isa<MemberExpr>(SimpleExpr)) {
    S.Diag(ELoc, diag::err_oss_expected_dereference_or_array_item)
        << RefExpr->getSourceRange();
    return;
  }
  while (auto *OASE = dyn_cast<OSSArraySectionExpr>(SimpleExpr)) {
    if (!OASE->isColonForm() && !OSSSyntax) {
      S.Diag(OASE->getColonLoc(), diag::err_oss_section_invalid_form)
          << RefExpr->getSourceRange();
      return;
    }
    SimpleExpr = OASE->getBase()->IgnoreParenCasts();
  }
}

Sema::DeclGroupPtrTy Sema::ActOnOmpSsDeclareTaskDirective(
    DeclGroupPtrTy DG,
    Expr *If, Expr *Final, Expr *Cost, Expr *Priority,
    ArrayRef<Expr *> Ins, ArrayRef<Expr *> Outs, ArrayRef<Expr *> Inouts,
    ArrayRef<Expr *> Concurrents, ArrayRef<Expr *> Commutatives,
    ArrayRef<Expr *> WeakIns, ArrayRef<Expr *> WeakOuts,
    ArrayRef<Expr *> WeakInouts,
    ArrayRef<Expr *> DepIns, ArrayRef<Expr *> DepOuts, ArrayRef<Expr *> DepInouts,
    ArrayRef<Expr *> DepConcurrents, ArrayRef<Expr *> DepCommutatives,
    ArrayRef<Expr *> DepWeakIns, ArrayRef<Expr *> DepWeakOuts,
    ArrayRef<Expr *> DepWeakInouts, SourceRange SR) {
  if (!DG || DG.get().isNull())
    return DeclGroupPtrTy();

  if (!DG.get().isSingleDecl()) {
    Diag(SR.getBegin(), diag::err_oss_single_decl_in_task);
    return DG;
  }
  Decl *ADecl = DG.get().getSingleDecl();
  if (auto *FTD = dyn_cast<FunctionTemplateDecl>(ADecl))
    ADecl = FTD->getTemplatedDecl();

  auto *FD = dyn_cast<FunctionDecl>(ADecl);
  if (!FD) {
    Diag(ADecl->getLocation(), diag::err_oss_function_expected);
    return DeclGroupPtrTy();
  }
  if (const CXXMethodDecl *MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->isVirtual() || isa<CXXConstructorDecl>(MD)
        || isa<CXXDestructorDecl>(MD)
        || MD->isOverloadedOperator()) {
      Diag(ADecl->getLocation(), diag::err_oss_function_expected) << 1;
      return DeclGroupPtrTy();
    }
  }
  if (FD->getReturnType() != Context.VoidTy) {
    Diag(ADecl->getLocation(), diag::err_oss_non_void_task);
    return DeclGroupPtrTy();
  }

  auto ParI = FD->param_begin();
  while (ParI != FD->param_end()) {
    QualType Type = (*ParI)->getType();
    if (!Type->isDependentType()
        && !Type.isPODType(Context)
        && !Type->isReferenceType()) {
      Diag((*ParI)->getBeginLoc(), diag::err_oss_non_pod_parm_task);
    }
    ++ParI;
  }

  ExprResult IfRes, FinalRes, CostRes, PriorityRes;
  if (If) {
    IfRes = VerifyBooleanConditionWithCleanups(If, If->getExprLoc());
  }
  if (Final) {
    FinalRes = VerifyBooleanConditionWithCleanups(Final, Final->getExprLoc());
  }
  if (Cost) {
    CostRes = CheckNonNegativeIntegerValue(
      Cost, OSSC_cost, /*StrictlyPositive=*/false);
  }
  if (Priority) {
    PriorityRes = CheckSignedIntegerValue(Priority);
  }
  for (Expr *RefExpr : Ins) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Outs) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Inouts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Concurrents) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : Commutatives) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakIns) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakOuts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : WeakInouts) {
    checkOutlineDependency(*this, RefExpr, /*OSSSyntax=*/true);
  }
  for (Expr *RefExpr : DepIns) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepOuts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepInouts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepConcurrents) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepCommutatives) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakIns) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakOuts) {
    checkOutlineDependency(*this, RefExpr);
  }
  for (Expr *RefExpr : DepWeakInouts) {
    checkOutlineDependency(*this, RefExpr);
  }

  auto *NewAttr = OSSTaskDeclAttr::CreateImplicit(
    Context,
    IfRes.get(), FinalRes.get(), CostRes.get(), PriorityRes.get(),
    const_cast<Expr **>(Ins.data()), Ins.size(),
    const_cast<Expr **>(Outs.data()), Outs.size(),
    const_cast<Expr **>(Inouts.data()), Inouts.size(),
    const_cast<Expr **>(Concurrents.data()), Concurrents.size(),
    const_cast<Expr **>(Commutatives.data()), Commutatives.size(),
    const_cast<Expr **>(WeakIns.data()), WeakIns.size(),
    const_cast<Expr **>(WeakOuts.data()), WeakOuts.size(),
    const_cast<Expr **>(WeakInouts.data()), WeakInouts.size(),
    const_cast<Expr **>(DepIns.data()), DepIns.size(),
    const_cast<Expr **>(DepOuts.data()), DepOuts.size(),
    const_cast<Expr **>(DepInouts.data()), DepInouts.size(),
    const_cast<Expr **>(DepConcurrents.data()), DepConcurrents.size(),
    const_cast<Expr **>(DepCommutatives.data()), DepCommutatives.size(),
    const_cast<Expr **>(DepWeakIns.data()), DepWeakIns.size(),
    const_cast<Expr **>(DepWeakOuts.data()), DepWeakOuts.size(),
    const_cast<Expr **>(DepWeakInouts.data()), DepWeakInouts.size(),
    SR);
  ADecl->addAttr(NewAttr);
  return DG;
}

// the boolean marks if it's a template
static std::pair<ValueDecl *, bool>
getPrivateItem(Sema &S, Expr *&RefExpr, SourceLocation &ELoc,
               SourceRange &ERange,
               bool AllowArraySection = false) {
  if (RefExpr->containsUnexpandedParameterPack()) {
    S.Diag(RefExpr->getExprLoc(), diag::err_oss_variadic_templates_not_clause_allowed);
    return std::make_pair(nullptr, false);
  } else if (RefExpr->isTypeDependent() || RefExpr->isValueDependent()) {
    return std::make_pair(nullptr, true);
  }

  RefExpr = RefExpr->IgnoreParens();
  enum {
    NoArrayExpr = -1,
    ArraySubscript = 0,
    OSSArraySection = 1
  } IsArrayExpr = NoArrayExpr;
  if (AllowArraySection) {
    if (auto *ASE = dyn_cast_or_null<ArraySubscriptExpr>(RefExpr)) {
      Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
      while (1) {
        // We can have:
        // 1) OSSArrayShapingExpr
        //    |- ArraySubscriptExpr
        // 2) ArraySubscriptExpr
        //    |- OSSArrayShapingExpr
        if (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
          Base = TempASE->getBase()->IgnoreParenImpCasts();
        else if (auto *TempOASE = dyn_cast<OSSArrayShapingExpr>(Base))
          Base = TempOASE->getBase()->IgnoreParenImpCasts();
        else
          break;
      }
      RefExpr = Base;
      IsArrayExpr = ArraySubscript;
    } else if (auto *OASE = dyn_cast_or_null<OSSArraySectionExpr>(RefExpr)) {
      Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
      while (auto *TempOASE = dyn_cast<OSSArraySectionExpr>(Base))
        Base = TempOASE->getBase()->IgnoreParenImpCasts();
      while (1) {
        // We can have:
        // 1) OSSArrayShapingExpr
        //    |- ArraySubscriptExpr
        // 2) ArraySubscriptExpr
        //    |- OSSArrayShapingExpr
        if (auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
          Base = TempASE->getBase()->IgnoreParenImpCasts();
        else if (auto *TempOASE = dyn_cast<OSSArrayShapingExpr>(Base))
          Base = TempOASE->getBase()->IgnoreParenImpCasts();
        else
          break;
      }
      RefExpr = Base;
      IsArrayExpr = OSSArraySection;
    }
  }

  ELoc = RefExpr->getExprLoc();
  ERange = RefExpr->getSourceRange();
  RefExpr = RefExpr->IgnoreParenImpCasts();
  auto *DE = dyn_cast_or_null<DeclRefExpr>(RefExpr);
  auto *ME = dyn_cast_or_null<MemberExpr>(RefExpr);

  // Only allow VarDecl from DeclRefExpr
  // and VarDecl implicits from MemberExpr // (i.e. static members without 'this')
  if ((!DE || !isa<VarDecl>(DE->getDecl())) &&
      (S.getCurrentThisType().isNull() || !ME ||
       !isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts()) ||
       !cast<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts())->isImplicit() ||
       !isa<VarDecl>(ME->getMemberDecl()))) {
    if (IsArrayExpr != NoArrayExpr) {
      // int *get();
      // reduction(+ : get()[3])
      S.Diag(ELoc, diag::err_oss_expected_base_var_name) << IsArrayExpr
                                                         << ERange;
    } else {
      S.Diag(ELoc,
             AllowArraySection
                 ? diag::err_oss_expected_var_name_member_expr_or_array_item
                 : diag::err_oss_expected_var_name_member_expr)
          << (S.getCurrentThisType().isNull() ? 0 : 1) << ERange;
    }
    return std::make_pair(nullptr, false);
  }

  auto *VD = cast<VarDecl>(DE ? DE->getDecl() : ME->getMemberDecl());

  return std::make_pair(getCanonicalDecl(VD), false);
}

bool Sema::ActOnOmpSsDependKinds(ArrayRef<OmpSsDependClauseKind> DepKinds,
                                 SmallVectorImpl<OmpSsDependClauseKind> &DepKindsOrdered,
                                 SourceLocation DepLoc) {
  if (DepKinds.size() == 2) {
    int numWeaks = 0;
    int numUnk = 0;
    // This is for concurrent and commutative. For now we don't allow combination
    // with 'weak'
    int numNoWeakCompats = 0;
    if (DepKinds[0] == OSSC_DEPEND_mutexinoutset
        || DepKinds[0] == OSSC_DEPEND_inoutset)
      ++numNoWeakCompats;
    if (DepKinds[1] == OSSC_DEPEND_mutexinoutset
        || DepKinds[1] == OSSC_DEPEND_inoutset)
      ++numNoWeakCompats;

    if (DepKinds[0] == OSSC_DEPEND_weak)
      ++numWeaks;
    else if (DepKinds[0] == OSSC_DEPEND_unknown)
      ++numUnk;
    if (DepKinds[1] == OSSC_DEPEND_weak)
      ++numWeaks;
    else if (DepKinds[1] == OSSC_DEPEND_unknown)
      ++numUnk;

    // concurrent/commutative cannot be combined with other modifiers
    if (numNoWeakCompats) {
        SmallString<256> Buffer;
        llvm::raw_svector_ostream Out(Buffer);
        Out << "'" << getOmpSsSimpleClauseTypeName(OSSC_depend, OSSC_DEPEND_mutexinoutset) << "'"
          << ", '" << getOmpSsSimpleClauseTypeName(OSSC_depend, OSSC_DEPEND_inoutset) << "'";
        Diag(DepLoc, diag::err_oss_depend_no_weak_compatible)
          << Out.str();
        return false;
    }

    if (numWeaks == 0) {
      if (numUnk == 0 || numUnk == 1) {
        Diag(DepLoc, diag::err_oss_depend_weak_required);
        return false;
      } else if (numUnk == 2) {
        unsigned Except[] = {OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
      }
    } else if ((numWeaks == 1 && numUnk == 1)
               || (numWeaks == 2 && numUnk == 0)) {
        unsigned Except[] = {OSSC_DEPEND_weak, OSSC_DEPEND_inoutset, OSSC_DEPEND_mutexinoutset};
        Diag(DepLoc, diag::err_oss_unexpected_clause_value)
            << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                       /*Last=*/OSSC_DEPEND_unknown, Except)
            << getOmpSsClauseName(OSSC_depend);
        return false;
    }
  } else {
    if (DepKinds[0] == OSSC_DEPEND_unknown
        || DepKinds[0] == OSSC_DEPEND_weak) {
      unsigned Except[] = {OSSC_DEPEND_weak};
      Diag(DepLoc, diag::err_oss_unexpected_clause_value)
          << getListOfPossibleValues(OSSC_depend, /*First=*/0,
                                     /*Last=*/OSSC_DEPEND_unknown, Except)
          << getOmpSsClauseName(OSSC_depend);
      return false;
    }
  }
  // Here we have three cases:
  // { OSSC_DEPEND_in }
  // { OSSC_DEPEND_weak, OSSC_DEPEND_in }
  // { OSSC_DEPEND_in, OSSC_DEPEND_weak }
  if (DepKinds[0] == OSSC_DEPEND_weak) {
    DepKindsOrdered.push_back(DepKinds[1]);
    DepKindsOrdered.push_back(DepKinds[0]);
  } else {
    DepKindsOrdered.push_back(DepKinds[0]);
    if (DepKinds.size() == 2)
      DepKindsOrdered.push_back(DepKinds[1]);
  }
  return true;
}

namespace {
/// Data for the reduction-based clauses.
struct ReductionData {
  /// List of simple vars of the reduction items. (data-sharings)
  SmallVector<Expr *, 8> SimpleVars;
  /// List of original reduction items.
  SmallVector<Expr *, 8> Vars;
  /// List of private copies of the reduction items.
  SmallVector<Expr *, 8> Privates;
  /// LHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> LHSs;
  /// RHS expressions for the reduction_op expressions.
  SmallVector<Expr *, 8> RHSs;
  /// Reduction operation expression.
  SmallVector<Expr *, 8> ReductionOps;
  /// Taskgroup descriptors for the corresponding reduction items in
  /// in_reduction clauses.
  SmallVector<Expr *, 8> TaskgroupDescriptors;
  /// List of captures for clause.
  SmallVector<Decl *, 4> ExprCaptures;
  /// List of postupdate expressions.
  SmallVector<Expr *, 4> ExprPostUpdates;
  ReductionData() = delete;
  /// Reserves required memory for the reduction data.
  ReductionData(unsigned Size) {
    SimpleVars.reserve(Size);
    Vars.reserve(Size);
    Privates.reserve(Size);
    LHSs.reserve(Size);
    RHSs.reserve(Size);
    ReductionOps.reserve(Size);
    TaskgroupDescriptors.reserve(Size);
    ExprCaptures.reserve(Size);
    ExprPostUpdates.reserve(Size);
  }
  /// Stores reduction item and reduction operation only (required for dependent
  /// reduction item).
  void push(Expr *Item, Expr *ReductionOp) {
    SimpleVars.emplace_back(nullptr);
    Vars.emplace_back(Item);
    Privates.emplace_back(nullptr);
    LHSs.emplace_back(nullptr);
    RHSs.emplace_back(nullptr);
    ReductionOps.emplace_back(ReductionOp);
    TaskgroupDescriptors.emplace_back(nullptr);
  }
  /// Stores reduction data.
  void push(Expr *SimpleVar, Expr *Item, Expr *Private, Expr *LHS, Expr *RHS, Expr *ReductionOp,
            Expr *TaskgroupDescriptor) {
    SimpleVars.emplace_back(SimpleVar);
    Vars.emplace_back(Item);
    Privates.emplace_back(Private);
    LHSs.emplace_back(LHS);
    RHSs.emplace_back(RHS);
    ReductionOps.emplace_back(ReductionOp);
    TaskgroupDescriptors.emplace_back(TaskgroupDescriptor);
  }
};
} // namespace

static bool actOnOSSReductionKindClause(
    Sema &S, OmpSsClauseKind ClauseKind,
    ArrayRef<Expr *> VarList, SourceLocation StartLoc, SourceLocation LParenLoc,
    SourceLocation ColonLoc, SourceLocation EndLoc,
    CXXScopeSpec &ReductionIdScopeSpec, const DeclarationNameInfo &ReductionId,
    ReductionData &RD) {
  DeclarationName DN = ReductionId.getName();
  OverloadedOperatorKind OOK = DN.getCXXOverloadedOperator();
  BinaryOperatorKind BOK = BO_Comma;

  ASTContext &Context = S.Context;
  // OmpSs-2 [2.14.3.6, reduction clause]
  // C
  // reduction-identifier is either an identifier or one of the following
  // operators: +, -, *,  &, |, ^, && and ||
  // C++
  // reduction-identifier is either an id-expression or one of the following
  // operators: +, -, *, &, |, ^, && and ||
  switch (OOK) {
  case OO_Plus:
  case OO_Minus:
    BOK = BO_Add;
    break;
  case OO_Star:
    BOK = BO_Mul;
    break;
  case OO_Amp:
    BOK = BO_And;
    break;
  case OO_Pipe:
    BOK = BO_Or;
    break;
  case OO_Caret:
    BOK = BO_Xor;
    break;
  case OO_AmpAmp:
    BOK = BO_LAnd;
    break;
  case OO_PipePipe:
    BOK = BO_LOr;
    break;
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Slash:
  case OO_Percent:
  case OO_Tilde:
  case OO_Exclaim:
  case OO_Equal:
  case OO_Less:
  case OO_Greater:
  case OO_LessEqual:
  case OO_GreaterEqual:
  case OO_PlusEqual:
  case OO_MinusEqual:
  case OO_StarEqual:
  case OO_SlashEqual:
  case OO_PercentEqual:
  case OO_CaretEqual:
  case OO_AmpEqual:
  case OO_PipeEqual:
  case OO_LessLess:
  case OO_GreaterGreater:
  case OO_LessLessEqual:
  case OO_GreaterGreaterEqual:
  case OO_EqualEqual:
  case OO_ExclaimEqual:
  case OO_Spaceship:
  case OO_PlusPlus:
  case OO_MinusMinus:
  case OO_Comma:
  case OO_ArrowStar:
  case OO_Arrow:
  case OO_Call:
  case OO_Subscript:
  case OO_Conditional:
  case OO_Coawait:
  case NUM_OVERLOADED_OPERATORS:
    llvm_unreachable("Unexpected reduction identifier");
  case OO_None:
    if (IdentifierInfo *II = DN.getAsIdentifierInfo()) {
      if (II->isStr("max"))
        BOK = BO_GT;
      else if (II->isStr("min"))
        BOK = BO_LT;
    }
    break;
  }
  SourceRange ReductionIdRange;
  if (ReductionIdScopeSpec.isValid())
    ReductionIdRange.setBegin(ReductionIdScopeSpec.getBeginLoc());
  else
    ReductionIdRange.setBegin(ReductionId.getBeginLoc());
  ReductionIdRange.setEnd(ReductionId.getEndLoc());

  for (Expr *RefExpr : VarList) {
    assert(RefExpr && "nullptr expr in OmpSs reduction clause.");
    // OmpSs [2.1, C/C++]
    //  A list item is a variable or array section, subject to the restrictions
    //  specified in Section 2.4 on page 42 and in each of the sections
    // describing clauses and directives for which a list appears.
    // OmpSs  [2.14.3.3, Restrictions, p.1]
    //  A variable that is part of another variable (as an array or
    //  structure element) cannot appear in a private clause.
    SourceLocation ELoc;
    SourceRange ERange;
    Expr *SimpleRefExpr = RefExpr;
    auto Res = getPrivateItem(S, SimpleRefExpr, ELoc, ERange,
                              /*AllowArraySection=*/false);
    if (Res.second) {
      // It will be analyzed later.
      Expr *ReductionOp = nullptr;
      RD.push(RefExpr, ReductionOp);
    }
    ValueDecl *D = Res.first;
    if (!D)
      continue;

    // Non-POD are not allowed in reductions
    if (!D->getType().isPODType(S.Context)) {
      S.Diag(ELoc, diag::err_oss_non_pod_reduction);
      continue;
    }

    QualType Type;
    auto *ASE = dyn_cast<ArraySubscriptExpr>(RefExpr->IgnoreParens());
    auto *OASE = dyn_cast<OSSArraySectionExpr>(RefExpr->IgnoreParens());
    if (ASE) {
      // TODO: For now we're not going to handle this because of AllowArraySection=false
      Type = ASE->getType().getNonReferenceType();
    } else if (OASE) {
      // TODO: For now we're not going to handle this because of AllowArraySection=false
      QualType BaseType =
          OSSArraySectionExpr::getBaseOriginalType(OASE->getBase());
      // Get the type of the base so we can handle the pointer case
      if (const auto *ATy = BaseType->getAsArrayTypeUnsafe())
        Type = ATy->getElementType();
      else
        Type = BaseType->getPointeeType();
      Type = Type.getNonReferenceType();
    } else {
      Type = Context.getBaseElementType(D->getType().getNonReferenceType());
    }
    auto *VD = dyn_cast<VarDecl>(D);

    // OmpSs [2.9.3.3, Restrictions, C/C++, p.3]
    //  A variable that appears in a private clause must not have an incomplete
    //  type or a reference type.
    if (S.RequireCompleteType(ELoc, D->getType(),
                              diag::err_oss_incomplete_type))
      continue;
    // OmpSs [2.14.3.6, reduction clause, Restrictions]
    // A list item that appears in a reduction clause must not be
    // const-qualified.
    // if (rejectConstNotMutableType(S, D, Type, ClauseKind, ELoc,
    //                               /*AcceptIfMutable*/ false, ASE || OASE))
    //   continue;

    Type = Type.getNonLValueExprType(Context).getUnqualifiedType();
    VarDecl *LHSVD = buildVarDecl(S, ELoc, Type, ".reduction.lhs",
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);
    VarDecl *RHSVD = buildVarDecl(S, ELoc, Type, D->getName(),
                                  D->hasAttrs() ? &D->getAttrs() : nullptr);

    // Private copy.
    VarDecl *PrivateVD =
        buildVarDecl(S, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
        // buildVarDecl(S, ELoc, PrivateTy, D->getName(),
        //              D->hasAttrs() ? &D->getAttrs() : nullptr,
        //              VD ? cast<DeclRefExpr>(SimpleRefExpr) : nullptr);

    // Add initializer for private variable.
    Expr *Init = nullptr;
    DeclRefExpr *LHSDRE = buildDeclRefExpr(S, LHSVD, Type, ELoc);
    DeclRefExpr *RHSDRE = buildDeclRefExpr(S, RHSVD, Type, ELoc);
    switch (BOK) {
    case BO_Add:
    case BO_Xor:
    case BO_Or:
    case BO_LOr:
      // '+', '-', '^', '|', '||' reduction ops - initializer is '0'.
      if (Type->isScalarType() || Type->isAnyComplexType())
        Init = S.ActOnIntegerConstant(ELoc, /*Val=*/0).get();
      break;
    case BO_Mul:
    case BO_LAnd:
      if (Type->isScalarType() || Type->isAnyComplexType()) {
        // '*' and '&&' reduction ops - initializer is '1'.
        Init = S.ActOnIntegerConstant(ELoc, /*Val=*/1).get();
      }
      break;
    case BO_And: {
      // '&' reduction op - initializer is '~0'.
      QualType OrigType = Type;
      if (auto *ComplexTy = OrigType->getAs<ComplexType>())
        Type = ComplexTy->getElementType();
      if (Type->isRealFloatingType()) {
        llvm::APFloat InitValue =
            llvm::APFloat::getAllOnesValue(Context.getTypeSize(Type),
                                           /*isIEEE=*/true);
        Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                       Type, ELoc);
      } else if (Type->isScalarType()) {
        uint64_t Size = Context.getTypeSize(Type);
        QualType IntTy = Context.getIntTypeForBitwidth(Size, /*Signed=*/0);
        llvm::APInt InitValue = llvm::APInt::getAllOnesValue(Size);
        Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
      }
      if (Init && OrigType->isAnyComplexType()) {
        // Init = 0xFFFF + 0xFFFFi;
        auto *Im = new (Context) ImaginaryLiteral(Init, OrigType);
        Init = S.CreateBuiltinBinOp(ELoc, BO_Add, Init, Im).get();
      }
      Type = OrigType;
      break;
    }
    case BO_LT:
    case BO_GT: {
      // 'min' reduction op - initializer is 'Largest representable number in
      // the reduction list item type'.
      // 'max' reduction op - initializer is 'Least representable number in
      // the reduction list item type'.
      if (Type->isIntegerType() || Type->isPointerType()) {
        bool IsSigned = Type->hasSignedIntegerRepresentation();
        uint64_t Size = Context.getTypeSize(Type);
        QualType IntTy =
            Context.getIntTypeForBitwidth(Size, /*Signed=*/IsSigned);
        llvm::APInt InitValue =
            (BOK != BO_LT) ? IsSigned ? llvm::APInt::getSignedMinValue(Size)
                                      : llvm::APInt::getMinValue(Size)
                           : IsSigned ? llvm::APInt::getSignedMaxValue(Size)
                                      : llvm::APInt::getMaxValue(Size);
        Init = IntegerLiteral::Create(Context, InitValue, IntTy, ELoc);
        if (Type->isPointerType()) {
          // Cast to pointer type.
          ExprResult CastExpr = S.BuildCStyleCastExpr(
              ELoc, Context.getTrivialTypeSourceInfo(Type, ELoc), ELoc, Init);
          if (CastExpr.isInvalid())
            continue;
          Init = CastExpr.get();
        }
      } else if (Type->isRealFloatingType()) {
        llvm::APFloat InitValue = llvm::APFloat::getLargest(
            Context.getFloatTypeSemantics(Type), BOK != BO_LT);
        Init = FloatingLiteral::Create(Context, InitValue, /*isexact=*/true,
                                       Type, ELoc);
      }
      break;
    }
    case BO_PtrMemD:
    case BO_PtrMemI:
    case BO_MulAssign:
    case BO_Div:
    case BO_Rem:
    case BO_Sub:
    case BO_Shl:
    case BO_Shr:
    case BO_LE:
    case BO_GE:
    case BO_EQ:
    case BO_NE:
    case BO_Cmp:
    case BO_AndAssign:
    case BO_XorAssign:
    case BO_OrAssign:
    case BO_Assign:
    case BO_AddAssign:
    case BO_SubAssign:
    case BO_DivAssign:
    case BO_RemAssign:
    case BO_ShlAssign:
    case BO_ShrAssign:
    case BO_Comma:
      llvm_unreachable("Unexpected reduction operation");
    }
    if (Init)
      S.AddInitializerToDecl(RHSVD, Init, /*DirectInit=*/false);
    else
      S.ActOnUninitializedDecl(RHSVD);
    if (RHSVD->isInvalidDecl())
      continue;
    if (!RHSVD->hasInit()) {
      // C structs do not have initializer
      S.Diag(ELoc, diag::err_oss_reduction_id_not_compatible)
          << Type << ReductionIdRange;
      bool IsDecl = !VD || VD->isThisDeclarationADefinition(Context) ==
                               VarDecl::DeclarationOnly;
      S.Diag(D->getLocation(),
             IsDecl ? diag::note_previous_decl : diag::note_defined_here)
          << D;
      continue;
    }
    // Store initializer for single element in private copy. Will be used during
    // codegen.
    PrivateVD->setInit(RHSVD->getInit());
    PrivateVD->setInitStyle(RHSVD->getInitStyle());
    DeclRefExpr *PrivateDRE = buildDeclRefExpr(S, PrivateVD, Type, ELoc);
    ExprResult ReductionOp;
    ReductionOp = S.BuildBinOp(
        nullptr /*Stack->getCurScope()*/, ReductionId.getBeginLoc(), BOK, LHSDRE, RHSDRE);
    if (ReductionOp.isUsable()) {
      if (BOK != BO_LT && BOK != BO_GT) {
        ReductionOp =
            S.BuildBinOp(nullptr /*Stack->getCurScope()*/, ReductionId.getBeginLoc(),
                         BO_Assign, LHSDRE, ReductionOp.get());
      } else {
        auto *ConditionalOp = new (Context)
            ConditionalOperator(ReductionOp.get(), ELoc, LHSDRE, ELoc, RHSDRE,
                                Type, VK_LValue, OK_Ordinary);
        ReductionOp =
            S.BuildBinOp(nullptr /*Stack->getCurScope()*/, ReductionId.getBeginLoc(),
                         BO_Assign, LHSDRE, ConditionalOp);
      }
      if (ReductionOp.isUsable())
        ReductionOp = S.ActOnFinishFullExpr(ReductionOp.get(),
                                            /*DiscardedValue*/ false);
    }
    if (!ReductionOp.isUsable())
      continue;

    RD.push(SimpleRefExpr, RefExpr->IgnoreParens(), PrivateDRE, LHSDRE, RHSDRE, ReductionOp.get(),
            /*TaskgroupDescriptor=*/nullptr);
  }
  return RD.Vars.empty();
}

OSSClause *
Sema::ActOnOmpSsReductionClause(ArrayRef<Expr *> VarList,
                       SourceLocation StartLoc, SourceLocation LParenLoc,
                       SourceLocation ColonLoc,
                       SourceLocation EndLoc,
                       CXXScopeSpec &ReductionIdScopeSpec,
                       const DeclarationNameInfo &ReductionId) {
  ReductionData RD(VarList.size());
  if (actOnOSSReductionKindClause(*this, OSSC_reduction, VarList,
                                  StartLoc, LParenLoc, ColonLoc, EndLoc,
                                  ReductionIdScopeSpec, ReductionId, RD))
    return nullptr;
  return OSSReductionClause::Create(
      Context, StartLoc, LParenLoc, ColonLoc, EndLoc, RD.Vars,
      ReductionIdScopeSpec.getWithLocInContext(Context), ReductionId,
      RD.SimpleVars, RD.Privates, RD.LHSs, RD.RHSs, RD.ReductionOps,
      /*buildPreInits(Context, RD.ExprCaptures)*/nullptr,
      /*buildPostUpdate(*this, RD.ExprPostUpdates)*/nullptr);
}

OSSClause *
Sema::ActOnOmpSsDependClause(ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
                             SourceLocation ColonLoc, ArrayRef<Expr *> VarList,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc, SourceLocation EndLoc,
                             bool OSSSyntax) {
  SmallVector<OmpSsDependClauseKind, 2> DepKindsOrdered;
  SmallVector<Expr *, 8> ClauseVars;
  if (!ActOnOmpSsDependKinds(DepKinds, DepKindsOrdered, DepLoc))
    return nullptr;

  for (Expr *RefExpr : VarList) {
    SourceLocation ELoc = RefExpr->getExprLoc();
    Expr *SimpleExpr = RefExpr->IgnoreParenCasts();
    if (RefExpr->containsUnexpandedParameterPack()) {
      Diag(RefExpr->getExprLoc(), diag::err_oss_variadic_templates_not_clause_allowed);
      continue;
    } else if (RefExpr->isTypeDependent() || RefExpr->isValueDependent()) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      continue;
    }

    if (RequireCompleteExprType(RefExpr, diag::err_oss_incomplete_type))
      continue;

    // TODO: check with OSSArraySectionExpr
    auto *ASE = dyn_cast<ArraySubscriptExpr>(SimpleExpr);
    // Allow only LValues, forbid ArraySubscripts over things
    // that are not an array like:
    //   typedef float V __attribute__((vector_size(16)));
    //   V a;
    //   #pragma oss task in(a[3])
    if (!RefExpr->IgnoreParenImpCasts()->isLValue() ||
        (ASE &&
         !ASE->getBase()->getType().getNonReferenceType()->isPointerType() &&
         !ASE->getBase()->getType().getNonReferenceType()->isArrayType())) {
      Diag(ELoc, diag::err_oss_expected_addressable_lvalue_or_array_item)
          << RefExpr->getSourceRange();
      continue;
    }
    bool InvalidArraySection = false;
    while (auto *OASE = dyn_cast<OSSArraySectionExpr>(SimpleExpr)) {
      if (!OASE->isColonForm() && !OSSSyntax) {
        Diag(OASE->getColonLoc(), diag::err_oss_section_invalid_form)
            << RefExpr->getSourceRange();
        // Only diagnose the first error
        InvalidArraySection = true;
        break;
      }
      SimpleExpr = OASE->getBase()->IgnoreParenImpCasts();
    }
    if (InvalidArraySection)
      continue;
    ClauseVars.push_back(RefExpr);
  }
  return OSSDependClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                 DepKinds, DepKindsOrdered,
                                 DepLoc, ColonLoc, ClauseVars,
                                 OSSSyntax);
}

OSSClause *
Sema::ActOnOmpSsVarListClause(
  OmpSsClauseKind Kind, ArrayRef<Expr *> Vars,
  SourceLocation StartLoc, SourceLocation LParenLoc,
  SourceLocation ColonLoc, SourceLocation EndLoc,
  ArrayRef<OmpSsDependClauseKind> DepKinds, SourceLocation DepLoc,
  CXXScopeSpec &ReductionIdScopeSpec,
  DeclarationNameInfo &ReductionId) {

  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_shared:
    Res = ActOnOmpSsSharedClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_private:
    Res = ActOnOmpSsPrivateClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_firstprivate:
    Res = ActOnOmpSsFirstprivateClause(Vars, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_depend:
    Res = ActOnOmpSsDependClause(DepKinds, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_reduction:
    Res = ActOnOmpSsReductionClause(Vars, StartLoc, LParenLoc, ColonLoc, EndLoc,
                                    ReductionIdScopeSpec, ReductionId);
    break;
  case OSSC_in:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_in }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_out:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_out }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_inout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inout }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_concurrent:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_mutexinoutset }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_commutative:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inoutset }, DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakin:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_in, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_out, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  case OSSC_weakinout:
    Res = ActOnOmpSsDependClause({ OSSC_DEPEND_inout, OSSC_DEPEND_weak },
                                 DepLoc, ColonLoc, Vars,
                                 StartLoc, LParenLoc, EndLoc, /*OSSSyntax=*/true);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }

  return Res;
}

OSSClause *
Sema::ActOnOmpSsSimpleClause(OmpSsClauseKind Kind,
                             unsigned Argument,
                             SourceLocation ArgumentLoc,
                             SourceLocation StartLoc,
                             SourceLocation LParenLoc,
                             SourceLocation EndLoc) {
  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_default:
    Res =
    ActOnOmpSsDefaultClause(static_cast<OmpSsDefaultClauseKind>(Argument),
                                 ArgumentLoc, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}

OSSClause *Sema::ActOnOmpSsDefaultClause(OmpSsDefaultClauseKind Kind,
                                          SourceLocation KindKwLoc,
                                          SourceLocation StartLoc,
                                          SourceLocation LParenLoc,
                                          SourceLocation EndLoc) {
  switch (Kind) {
  case OSSC_DEFAULT_none:
    DSAStack->setDefaultDSANone(KindKwLoc);
    break;
  case OSSC_DEFAULT_shared:
    DSAStack->setDefaultDSAShared(KindKwLoc);
    break;
  case OSSC_DEFAULT_unknown:
    Diag(KindKwLoc, diag::err_oss_unexpected_clause_value)
        << getListOfPossibleValues(OSSC_default, /*First=*/0,
                                   /*Last=*/OSSC_DEFAULT_unknown)
        << getOmpSsClauseName(OSSC_default);
    return nullptr;
  }
  return new (Context)
      OSSDefaultClause(Kind, KindKwLoc, StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::PerformOmpSsImplicitIntegerConversion(SourceLocation Loc,
                                                       Expr *Op) {
  if (!Op)
    return ExprError();

  class IntConvertDiagnoser : public ICEConvertDiagnoser {
  public:
    IntConvertDiagnoser()
        : ICEConvertDiagnoser(/*AllowScopedEnumerations*/ false, false, true) {}
    SemaDiagnosticBuilder diagnoseNotInt(Sema &S, SourceLocation Loc,
                                         QualType T) override {
      return S.Diag(Loc, diag::err_oss_not_integral) << T;
    }
    SemaDiagnosticBuilder diagnoseIncomplete(Sema &S, SourceLocation Loc,
                                             QualType T) override {
      return S.Diag(Loc, diag::err_oss_incomplete_type) << T;
    }
    SemaDiagnosticBuilder diagnoseExplicitConv(Sema &S, SourceLocation Loc,
                                               QualType T,
                                               QualType ConvTy) override {
      return S.Diag(Loc, diag::err_oss_explicit_conversion) << T << ConvTy;
    }
    SemaDiagnosticBuilder noteExplicitConv(Sema &S, CXXConversionDecl *Conv,
                                           QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_oss_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseAmbiguous(Sema &S, SourceLocation Loc,
                                            QualType T) override {
      return S.Diag(Loc, diag::err_oss_ambiguous_conversion) << T;
    }
    SemaDiagnosticBuilder noteAmbiguous(Sema &S, CXXConversionDecl *Conv,
                                        QualType ConvTy) override {
      return S.Diag(Conv->getLocation(), diag::note_oss_conversion_here)
             << ConvTy->isEnumeralType() << ConvTy;
    }
    SemaDiagnosticBuilder diagnoseConversion(Sema &, SourceLocation, QualType,
                                             QualType) override {
      llvm_unreachable("conversion functions are permitted");
    }
  } ConvertDiagnoser;
  return PerformContextualImplicitConversion(Loc, Op, ConvertDiagnoser);
}


OSSClause *
Sema::ActOnOmpSsSharedClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc,
                       bool isImplicit) {
  SmallVector<Expr *, 8> ClauseVars;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;
    // Implicit CXXThisExpr generated by the compiler are fine
    if (isImplicit && isa<CXXThisExpr>(RefExpr)) {
      ClauseVars.push_back(RefExpr);
      continue;
    }

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
    }
    ValueDecl *D = Res.first;
    if (!D) {
      continue;
    }

    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_shared &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_shared);
      continue;
    }
    DSAStack->addDSA(D, RefExpr, OSSC_shared, /*Ignore=*/false, /*Implicit=*/false);
    ClauseVars.push_back(RefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSSharedClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars);
}

OSSClause *
Sema::ActOnOmpSsPrivateClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 8> ClauseVars;
  SmallVector<Expr *, 8> PrivateCopies;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D) {
      continue;
    }

    if (RequireCompleteType(ELoc, D->getType(),
                            diag::err_oss_incomplete_type))
      continue;

    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_private &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_private);
      continue;
    }

    QualType Type = D->getType().getUnqualifiedType().getNonReferenceType();
    if (Type->isArrayType())
      Type = Context.getBaseElementType(Type).getCanonicalType();

    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OmpSs-2 region uses
    // original variable for proper diagnostics and variable capturing.

    // Build DSA Copy
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    ActOnUninitializedDecl(VDPrivate);

    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, Type, RefExpr->getExprLoc());

    DSAStack->addDSA(D, RefExpr, OSSC_private, /*Ignore=*/false, /*Implicit=*/false);
    ClauseVars.push_back(RefExpr);
    PrivateCopies.push_back(VDPrivateRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSPrivateClause::Create(Context, StartLoc, LParenLoc, EndLoc, ClauseVars, PrivateCopies);
}

OSSClause *
Sema::ActOnOmpSsFirstprivateClause(ArrayRef<Expr *> Vars,
                       SourceLocation StartLoc,
                       SourceLocation LParenLoc,
                       SourceLocation EndLoc) {
  SmallVector<Expr *, 8> ClauseVars;
  SmallVector<Expr *, 8> PrivateCopies;
  SmallVector<Expr *, 8> Inits;
  for (Expr *RefExpr : Vars) {

    SourceLocation ELoc;
    SourceRange ERange;

    auto Res = getPrivateItem(*this, RefExpr, ELoc, ERange);
    if (Res.second) {
      // It will be analyzed later.
      ClauseVars.push_back(RefExpr);
      PrivateCopies.push_back(nullptr);
      Inits.push_back(nullptr);
    }
    ValueDecl *D = Res.first;
    if (!D) {
      continue;
    }

    if (RequireCompleteType(ELoc, D->getType(),
                            diag::err_oss_incomplete_type))
      continue;

    DSAStackTy::DSAVarData DVar = DSAStack->getCurrentDSA(D);
    if (DVar.CKind != OSSC_unknown && DVar.CKind != OSSC_firstprivate &&
        DVar.RefExpr) {
      Diag(ELoc, diag::err_oss_wrong_dsa) << getOmpSsClauseName(DVar.CKind)
                                          << getOmpSsClauseName(OSSC_firstprivate);
      continue;
    }

    QualType Type = D->getType().getUnqualifiedType().getNonReferenceType();
    if (Type->isArrayType())
      Type = Context.getBaseElementType(Type).getCanonicalType();

    // Generate helper private variable and initialize it with the value of the
    // original variable. The address of the original variable is replaced by
    // the address of the new private variable in the CodeGen. This new variable
    // is not added to IdResolver, so the code in the OmpSs-2 region uses
    // original variable for proper diagnostics and variable capturing.

    // Build DSA clone
    VarDecl *VDPrivate =
        buildVarDecl(*this, ELoc, Type, D->getName(),
                     D->hasAttrs() ? &D->getAttrs() : nullptr);
    Expr *VDInitRefExpr = nullptr;
    // Build a temp variable to use it as initializer
    VarDecl *VDInit = buildVarDecl(*this, RefExpr->getExprLoc(), Type,
                                   ".firstprivate.temp");
    VDInitRefExpr = buildDeclRefExpr(*this, VDInit, Type,
                                     RefExpr->getExprLoc());
    // Set temp variable as initializer of DSA clone
    AddInitializerToDecl(VDPrivate,
                         DefaultLvalueConversion(VDInitRefExpr).get(),
                         /*DirectInit=*/false);

    DeclRefExpr *VDPrivateRefExpr = buildDeclRefExpr(
        *this, VDPrivate, Type, RefExpr->getExprLoc());

    DSAStack->addDSA(D, RefExpr, OSSC_firstprivate, /*Ignore=*/false, /*Implicit=*/false);
    ClauseVars.push_back(RefExpr);
    PrivateCopies.push_back(VDPrivateRefExpr);
    Inits.push_back(VDInitRefExpr);
  }

  if (Vars.empty())
    return nullptr;

  return OSSFirstprivateClause::Create(Context, StartLoc, LParenLoc, EndLoc,
                                       ClauseVars, PrivateCopies, Inits);
}

ExprResult Sema::CheckNonNegativeIntegerValue(Expr *ValExpr,
                                      OmpSsClauseKind CKind,
                                      bool StrictlyPositive) {
  ExprResult Res = CheckSignedIntegerValue(ValExpr);
  if (Res.isInvalid())
    return ExprError();

  ValExpr = Res.get();
  // The expression must evaluate to a non-negative integer value.
  llvm::APSInt Result;
  if (ValExpr->isIntegerConstantExpr(Result, Context) &&
      Result.isSigned() &&
      !((!StrictlyPositive && Result.isNonNegative()) ||
        (StrictlyPositive && Result.isStrictlyPositive()))) {
    Diag(ValExpr->getExprLoc(), diag::err_oss_negative_expression_in_clause)
        << getOmpSsClauseName(CKind) << (StrictlyPositive ? 1 : 0)
        << ValExpr->getSourceRange();
    return ExprError();
  }
  return ValExpr;
}

ExprResult Sema::VerifyBooleanConditionWithCleanups(
    Expr *Condition,
    SourceLocation StartLoc) {

  if (!Condition->isValueDependent() && !Condition->isTypeDependent() &&
      !Condition->isInstantiationDependent() &&
      !Condition->containsUnexpandedParameterPack()) {
    ExprResult Val = CheckBooleanCondition(StartLoc, Condition);
    if (Val.isInvalid())
      return ExprError();

    return MakeFullExpr(Val.get()).get();
  }
  return Condition;
}

OSSClause *Sema::ActOnOmpSsIfClause(Expr *Condition,
                                    SourceLocation StartLoc,
                                    SourceLocation LParenLoc,
                                    SourceLocation EndLoc) {
  ExprResult Res = VerifyBooleanConditionWithCleanups(Condition, StartLoc);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSIfClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsFinalClause(Expr *Condition,
                                       SourceLocation StartLoc,
                                       SourceLocation LParenLoc,
                                       SourceLocation EndLoc) {
  ExprResult Res = VerifyBooleanConditionWithCleanups(Condition, StartLoc);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSFinalClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsCostClause(Expr *E,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc) {
  // The parameter of the cost() clause must be > 0
  // expression.
  ExprResult Res = CheckNonNegativeIntegerValue(
    E, OSSC_cost, /*StrictlyPositive=*/false);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSCostClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

ExprResult Sema::CheckSignedIntegerValue(Expr *ValExpr) {
  if (!ValExpr->isTypeDependent() && !ValExpr->isValueDependent() &&
      !ValExpr->isInstantiationDependent() &&
      !ValExpr->containsUnexpandedParameterPack()) {
    SourceLocation Loc = ValExpr->getExprLoc();
    ExprResult Value =
        PerformOmpSsImplicitIntegerConversion(Loc, ValExpr);
    if (Value.isInvalid())
      return ExprError();
    return Value.get();
  }
  return ExprEmpty();
}

OSSClause *Sema::ActOnOmpSsPriorityClause(Expr *E,
                                      SourceLocation StartLoc,
                                      SourceLocation LParenLoc,
                                      SourceLocation EndLoc) {
  // The parameter of the priority() clause must be integer signed
  // expression.
  ExprResult Res = CheckSignedIntegerValue(E);
  if (Res.isInvalid())
    return nullptr;

  return new (Context) OSSPriorityClause(Res.get(), StartLoc, LParenLoc, EndLoc);
}

OSSClause *Sema::ActOnOmpSsSingleExprClause(OmpSsClauseKind Kind, Expr *Expr,
                                            SourceLocation StartLoc,
                                            SourceLocation LParenLoc,
                                            SourceLocation EndLoc) {
  OSSClause *Res = nullptr;
  switch (Kind) {
  case OSSC_if:
    Res = ActOnOmpSsIfClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_final:
    Res = ActOnOmpSsFinalClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_cost:
    Res = ActOnOmpSsCostClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  case OSSC_priority:
    Res = ActOnOmpSsPriorityClause(Expr, StartLoc, LParenLoc, EndLoc);
    break;
  default:
    llvm_unreachable("Clause is not allowed.");
  }
  return Res;
}
