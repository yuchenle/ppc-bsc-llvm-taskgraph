//===----- CGOmpSsRuntime.h - Interface to OmpSs Runtimes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OmpSs runtime code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOMPSSRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGOMPSSRUNTIME_H

#include "CGCall.h"
#include "CGValue.h"
#include "clang/AST/Type.h"
#include "clang/Basic/OmpSsKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueHandle.h"

namespace clang {
class OSSExecutableDirective;

namespace CodeGen {
class Address;
class CodeGenFunction;
class CodeGenModule;

struct OSSDSAPrivateDataTy {
  const Expr *Ref;
  const Expr *Copy;
};

struct OSSDSAFirstprivateDataTy {
  const Expr *Ref;
  const Expr *Copy;
  const Expr *Init;
};

struct OSSTaskDSADataTy final {
  SmallVector<const Expr *, 4> Shareds;
  SmallVector<OSSDSAPrivateDataTy, 4> Privates;
  SmallVector<OSSDSAFirstprivateDataTy, 4> Firstprivates;

  bool empty() const {
    return Shareds.empty() && Privates.empty()
      && Firstprivates.empty();
  }
};

struct OSSDepDataTy {
  bool OSSSyntax;
  const Expr *E;
};

struct OSSTaskDepDataTy final {
  SmallVector<OSSDepDataTy, 4> WeakIns;
  SmallVector<OSSDepDataTy, 4> WeakOuts;
  SmallVector<OSSDepDataTy, 4> WeakInouts;
  SmallVector<OSSDepDataTy, 4> Ins;
  SmallVector<OSSDepDataTy, 4> Outs;
  SmallVector<OSSDepDataTy, 4> Inouts;
  SmallVector<OSSDepDataTy, 4> WeakConcurrents;
  SmallVector<OSSDepDataTy, 4> WeakCommutatives;
  SmallVector<OSSDepDataTy, 4> Concurrents;
  SmallVector<OSSDepDataTy, 4> Commutatives;

  bool empty() const {
  return WeakIns.empty() && WeakOuts.empty() && WeakInouts.empty()
    && Ins.empty() && Outs.empty() && Inouts.empty()
    && WeakConcurrents.empty() && WeakCommutatives.empty()
    && Concurrents.empty() && Commutatives.empty();
  }
};

struct OSSReductionDataTy {
  const Expr *Ref;
  const Expr *LHS;
  const Expr *RHS;
  const Expr *ReductionOp;
  const BinaryOperatorKind ReductionKind;
};

struct OSSTaskReductionDataTy final {
  SmallVector<OSSReductionDataTy, 4> RedList;
  SmallVector<OSSReductionDataTy, 4> WeakRedList;

  bool empty() const {
    return RedList.empty() && WeakRedList.empty();
  }
};

struct OSSTaskDataTy final {
  OSSTaskDSADataTy DSAs;
  OSSTaskDepDataTy Deps;
  OSSTaskReductionDataTy Reductions;
  const Expr *If = nullptr;
  const Expr *Final = nullptr;
  const Expr *Cost = nullptr;
  const Expr *Priority = nullptr;
  const Expr *Label = nullptr;
  bool Wait = false;

  bool empty() const {
    return DSAs.empty() && Deps.empty() &&
      Reductions.empty() &&
      !If && !Final && !Cost && !Priority;
  }
};

struct OSSLoopDataTy final {
  const Expr *IndVar = nullptr;
  const Expr *LB = nullptr;
  const Expr *UB = nullptr;
  const Expr *Step = nullptr;
  const Expr *Chunksize = nullptr;
  const Expr *Grainsize = nullptr;
  llvm::Optional<bool> TestIsLessOp;
  bool TestIsStrictOp;
  bool empty() const {
    return !IndVar &&
          !LB && !UB && !Step;
  }
};

class CGOmpSsRuntime {
protected:
  CodeGenModule &CGM;

private:
  struct TaskContext {
    llvm::AssertingVH<llvm::Instruction> InsertPt = nullptr;
    llvm::BasicBlock *TerminateLandingPad = nullptr;
    llvm::BasicBlock *TerminateHandler = nullptr;
    llvm::BasicBlock *UnreachableBlock = nullptr;
    Address ExceptionSlot = Address::invalid();
    Address EHSelectorSlot = Address::invalid();
    Address NormalCleanupDestSlot = Address::invalid();
  };

private:

  SmallVector<TaskContext, 2> TaskStack;

  // Map to reuse Addresses emited for data sharings
  using CaptureMapTy = llvm::DenseMap<const VarDecl *, Address>;
  SmallVector<CaptureMapTy, 2> CaptureMapStack;

  // Map of builtin reduction init/combiner <Nanos6 int value, <init, combiner>>
  using BuiltinRedMapTy = llvm::DenseMap<llvm::Value *, std::pair<llvm::Value *, llvm::Value *>>;
  BuiltinRedMapTy BuiltinRedMap;

  // Map of UDR init/combiner <UDR, <init, combiner>>
  using UDRMapTy = llvm::DenseMap<const OSSDeclareReductionDecl *, std::pair<llvm::Value *, llvm::Value *>>;
  UDRMapTy UDRMap;

  // This is used to avoid creating the same generic funcion for constructors and
  // destructors, which will be stored in a bundle for each non-pod private/firstprivate
  // data-sharing
  using GenericCXXNonPodMethodDefsTy = llvm::DenseMap<const CXXMethodDecl *, llvm::Function *>;
  GenericCXXNonPodMethodDefsTy GenericCXXNonPodMethodDefs;

  void EmitDSAShared(
    CodeGenFunction &CGF, const Expr *E,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
    SmallVectorImpl<llvm::Value*> &CapturedList);

  void EmitDSAPrivate(
    CodeGenFunction &CGF, const OSSDSAPrivateDataTy &PDataTy,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
    SmallVectorImpl<llvm::Value*> &CapturedList);

  void EmitDSAFirstprivate(
    CodeGenFunction &CGF, const OSSDSAFirstprivateDataTy &PDataTy,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
    SmallVectorImpl<llvm::Value*> &CapturedList);

  void EmitDependencyList(
    CodeGenFunction &CGF, const OSSDepDataTy &Dep,
    SmallVectorImpl<llvm::Value *> &List);

  void EmitDependency(
    std::string Name, CodeGenFunction &CGF, const OSSDepDataTy &Dep,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo);

  void EmitReduction(
    std::string RedName, std::string RedInitName, std::string RedCombName,
    CodeGenFunction &CGF, const OSSReductionDataTy &Red,
    SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo);

  void EmitCopyCtorFunc(llvm::Value *DSAValue, const CXXConstructExpr *CtorE,
      const VarDecl *CopyD, const VarDecl *InitD,
      SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo);

  void EmitCtorFunc(llvm::Value *DSAValue, const VarDecl *CopyD,
      SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo);

  void EmitDtorFunc(llvm::Value *DSAValue, const VarDecl *CopyD,
      SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo);

  // Build bundles for all info inside Data
  void EmitDirectiveData(CodeGenFunction &CGF, const OSSTaskDataTy &Data,
      SmallVectorImpl<llvm::OperandBundleDef> &TaskInfo,
      const OSSLoopDataTy &LoopData = OSSLoopDataTy());

public:
  explicit CGOmpSsRuntime(CodeGenModule &CGM) : CGM(CGM) {}
  virtual ~CGOmpSsRuntime() {};
  virtual void clear() {};

  bool InTaskEmission = false;

  // returns true if we're emitting code inside a task context (entry/exit)
  bool inTaskBody();
  // returns the innermost nested task InsertPt instruction
  llvm::AssertingVH<llvm::Instruction> getTaskInsertPt();
  // returns the innermost nested task TerminateHandler BB
  llvm::BasicBlock *getTaskTerminateHandler();
  // returns the innermost nested task TerminateLandingPad BB
  llvm::BasicBlock *getTaskTerminateLandingPad();
  // returns the innermost nested task UnreachableBlock BB
  llvm::BasicBlock *getTaskUnreachableBlock();
  // returns the innermost nested task ExceptionSlot address
  Address getTaskExceptionSlot();
  // returns the innermost nested task EHSelectorSlot address
  Address getTaskEHSelectorSlot();
  // returns the innermost nested task NormalCleanupDestSlot address
  Address getTaskNormalCleanupDestSlot();
  // returns the captured address of VD
  Address getTaskCaptureAddr(const VarDecl *VD);

  // sets the innermost nested task InsertPt instruction
  void setTaskInsertPt(llvm::Instruction *I);
  // sets the innermost nested task TerminateHandler instruction
  void setTaskTerminateHandler(llvm::BasicBlock *BB);
  // sets the innermost nested task TerminateLandingPad instruction
  void setTaskTerminateLandingPad(llvm::BasicBlock *BB);
  // sets the innermost nested task UnreachableBlock instruction
  void setTaskUnreachableBlock(llvm::BasicBlock *BB);
  // sets the innermost nested task ExceptionSlot address
  void setTaskExceptionSlot(Address Addr);
  // sets the innermost nested task EHSelectorSlot address
  void setTaskEHSelectorSlot(Address Addr);
  // returns the innermost nested task NormalCleanupDestSlot address
  void setTaskNormalCleanupDestSlot(Address Addr);

  RValue emitTaskFunction(CodeGenFunction &CGF,
                          const FunctionDecl *FD,
                          const CallExpr *CE,
                          ReturnValueSlot ReturnValue);

  /// Emit code for 'taskwait' directive.
  virtual void emitTaskwaitCall(CodeGenFunction &CGF,
                                SourceLocation Loc,
                                const OSSTaskDataTy &Data);
  /// Emit code for 'task' directive.
  virtual void emitTaskCall(CodeGenFunction &CGF,
                            const OSSExecutableDirective &D,
                            SourceLocation Loc,
                            const OSSTaskDataTy &Data);

  /// Emit code for 'task' directive.
  virtual void emitLoopCall(CodeGenFunction &CGF,
                            const OSSLoopDirective &D,
                            SourceLocation Loc,
                            const OSSTaskDataTy &Data,
                            const OSSLoopDataTy &LoopData);

};

} // namespace CodeGen
} // namespace clang

#endif
