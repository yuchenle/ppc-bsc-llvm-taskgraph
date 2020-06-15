//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
using namespace clang;

void LoopHintAttr::printPrettyPragma(raw_ostream &OS,
                                     const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  // For "#pragma unroll" and "#pragma nounroll" the string "unroll" or
  // "nounroll" is already emitted as the pragma name.
  if (SpellingIndex == Pragma_nounroll ||
      SpellingIndex == Pragma_nounroll_and_jam)
    return;
  else if (SpellingIndex == Pragma_unroll ||
           SpellingIndex == Pragma_unroll_and_jam) {
    OS << ' ' << getValueString(Policy);
    return;
  }

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  OS << ' ' << getOptionName(option) << getValueString(Policy);
}

// Return a string containing the loop hint argument including the
// enclosing parentheses.
std::string LoopHintAttr::getValueString(const PrintingPolicy &Policy) const {
  std::string ValueName;
  llvm::raw_string_ostream OS(ValueName);
  OS << "(";
  if (state == Numeric)
    value->printPretty(OS, nullptr, Policy);
  else if (state == Enable)
    OS << "enable";
  else if (state == Full)
    OS << "full";
  else if (state == AssumeSafety)
    OS << "assume_safety";
  else
    OS << "disable";
  OS << ")";
  return OS.str();
}

// Return a string suitable for identifying this attribute in diagnostics.
std::string
LoopHintAttr::getDiagnosticName(const PrintingPolicy &Policy) const {
  unsigned SpellingIndex = getAttributeSpellingListIndex();
  if (SpellingIndex == Pragma_nounroll)
    return "#pragma nounroll";
  else if (SpellingIndex == Pragma_unroll)
    return "#pragma unroll" +
           (option == UnrollCount ? getValueString(Policy) : "");
  else if (SpellingIndex == Pragma_nounroll_and_jam)
    return "#pragma nounroll_and_jam";
  else if (SpellingIndex == Pragma_unroll_and_jam)
    return "#pragma unroll_and_jam" +
           (option == UnrollAndJamCount ? getValueString(Policy) : "");

  assert(SpellingIndex == Pragma_clang_loop && "Unexpected spelling");
  return getOptionName(option) + getValueString(Policy);
}

void OMPDeclareSimdDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (getBranchState() != BS_Undefined)
    OS << ' ' << ConvertBranchStateTyToStr(getBranchState());
  if (auto *E = getSimdlen()) {
    OS << " simdlen(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (uniforms_size() > 0) {
    OS << " uniform";
    StringRef Sep = "(";
    for (auto *E : uniforms()) {
      OS << Sep;
      E->printPretty(OS, nullptr, Policy);
      Sep = ", ";
    }
    OS << ")";
  }
  alignments_iterator NI = alignments_begin();
  for (auto *E : aligneds()) {
    OS << " aligned(";
    E->printPretty(OS, nullptr, Policy);
    if (*NI) {
      OS << ": ";
      (*NI)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++NI;
  }
  steps_iterator I = steps_begin();
  modifiers_iterator MI = modifiers_begin();
  for (auto *E : linears()) {
    OS << " linear(";
    if (*MI != OMPC_LINEAR_unknown)
      OS << getOpenMPSimpleClauseTypeName(llvm::omp::Clause::OMPC_linear, *MI)
         << "(";
    E->printPretty(OS, nullptr, Policy);
    if (*MI != OMPC_LINEAR_unknown)
      OS << ")";
    if (*I) {
      OS << ": ";
      (*I)->printPretty(OS, nullptr, Policy);
    }
    OS << ")";
    ++I;
    ++MI;
  }
}

void OSSTaskDeclAttr::printPrettyPragma(
    raw_ostream & OS, const PrintingPolicy &Policy) const {

  auto l = [](StringRef S, StringRef Sep, unsigned Size, Expr **Begin, Expr **End,
              raw_ostream &OS, const PrintingPolicy &Policy) {
    if (Size > 0) {
      OS << " " << S;
      Expr **I = Begin;
      while (I != End) {
        OS << Sep;
        (*I)->printPretty(OS, nullptr, Policy);
        Sep = ", ";
        ++I;
      }
      OS << ")";
    }
  };
  l("in", "(", ins_size(), ins_begin(), ins_end(), OS, Policy);
  l("out", "(", outs_size(), outs_begin(), outs_end(), OS, Policy);
  l("inout", "(", inouts_size(), inouts_begin(), inouts_end(), OS, Policy);
  l("concurrent", "(", concurrents_size(), concurrents_begin(), concurrents_end(), OS, Policy);
  l("commutative", "(", commutatives_size(), commutatives_begin(), commutatives_end(), OS, Policy);
  l("weakin", "(", weakIns_size(), weakIns_begin(), weakIns_end(), OS, Policy);
  l("weakout", "(", weakOuts_size(), weakOuts_begin(), weakOuts_end(), OS, Policy);
  l("weakinout", "(", weakInouts_size(), weakInouts_begin(), weakInouts_end(), OS, Policy);
  l("weakconcurrent", "(", weakConcurrents_size(), weakConcurrents_begin(), weakConcurrents_end(), OS, Policy);
  l("weakcommutative", "(", weakCommutatives_size(), weakCommutatives_begin(), weakCommutatives_end(), OS, Policy);
  l("depend(in", ":", depIns_size(), depIns_begin(), depIns_end(), OS, Policy);
  l("depend(out", ":", depOuts_size(), depOuts_begin(), depOuts_end(), OS, Policy);
  l("depend(inout", ":", depInouts_size(), depInouts_begin(), depInouts_end(), OS, Policy);
  l("depend(inoutset", ":", depConcurrents_size(), depConcurrents_begin(), depConcurrents_end(), OS, Policy);
  l("depend(mutexinoutset", ":", depCommutatives_size(), depCommutatives_begin(), depCommutatives_end(), OS, Policy);
  l("depend(weak, in", ":", depWeakIns_size(), depWeakIns_begin(), depWeakIns_end(), OS, Policy);
  l("depend(weak, out", ":", depWeakOuts_size(), depWeakOuts_begin(), depWeakOuts_end(), OS, Policy);
  l("depend(weak, inout", ":", depWeakInouts_size(), depWeakInouts_begin(), depWeakInouts_end(), OS, Policy);
  l("depend(weak, inoutset", ":", depWeakConcurrents_size(), depWeakConcurrents_begin(), depWeakConcurrents_end(), OS, Policy);
  l("depend(weak, mutexinoutset", ":", depWeakCommutatives_size(), depWeakCommutatives_begin(), depWeakCommutatives_end(), OS, Policy);
  if (auto *E = getIfExpr()) {
    OS << " if(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (auto *E = getFinalExpr()) {
    OS << " final(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (auto *E = getCostExpr()) {
    OS << " cost(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (auto *E = getPriorityExpr()) {
    OS << "priority(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  if (getWait())
      OS << " wait";
}

void OMPDeclareTargetDeclAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  // Use fake syntax because it is for testing and debugging purpose only.
  if (getDevType() != DT_Any)
    OS << " device_type(" << ConvertDevTypeTyToStr(getDevType()) << ")";
  if (getMapType() != MT_To)
    OS << ' ' << ConvertMapTypeTyToStr(getMapType());
}

llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy>
OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(const ValueDecl *VD) {
  if (!VD->hasAttrs())
    return llvm::None;
  if (const auto *Attr = VD->getAttr<OMPDeclareTargetDeclAttr>())
    return Attr->getMapType();

  return llvm::None;
}

llvm::Optional<OMPDeclareTargetDeclAttr::DevTypeTy>
OMPDeclareTargetDeclAttr::getDeviceType(const ValueDecl *VD) {
  if (!VD->hasAttrs())
    return llvm::None;
  if (const auto *Attr = VD->getAttr<OMPDeclareTargetDeclAttr>())
    return Attr->getDevType();

  return llvm::None;
}

namespace clang {
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo &TI);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const OMPTraitInfo *TI);
}

void OMPDeclareVariantAttr::printPrettyPragma(
    raw_ostream &OS, const PrintingPolicy &Policy) const {
  if (const Expr *E = getVariantFuncRef()) {
    OS << "(";
    E->printPretty(OS, nullptr, Policy);
    OS << ")";
  }
  OS << " match(" << traitInfos << ")";
}

#include "clang/AST/AttrImpl.inc"
