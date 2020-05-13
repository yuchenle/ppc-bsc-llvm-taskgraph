// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump=json -ast-dump-filter Test %s | FileCheck %s

namespace Test {

namespace NS {
void Function();
}
void NS::Function() {}

struct S {
  void Method();
};
void S::Method() {}

} // namespace Test

// NOTE: CHECK lines have been autogenerated by gen_ast_dump_json_test.py


// CHECK:  "kind": "NamespaceDecl",
// CHECK-NEXT:  "loc": {
// CHECK-NEXT:   "offset": 116,
// CHECK-NEXT:   "file": "{{.*}}",
// CHECK-NEXT:   "line": 3,
// CHECK-NEXT:   "col": 11,
// CHECK-NEXT:   "tokLen": 4
// CHECK-NEXT:  },
// CHECK-NEXT:  "range": {
// CHECK-NEXT:   "begin": {
// CHECK-NEXT:    "offset": 106,
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "tokLen": 9
// CHECK-NEXT:   },
// CHECK-NEXT:   "end": {
// CHECK-NEXT:    "offset": 234,
// CHECK-NEXT:    "line": 15,
// CHECK-NEXT:    "col": 1,
// CHECK-NEXT:    "tokLen": 1
// CHECK-NEXT:   }
// CHECK-NEXT:  },
// CHECK-NEXT:  "name": "Test",
// CHECK-NEXT:  "inner": [
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "NamespaceDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "offset": 134,
// CHECK-NEXT:     "line": 5,
// CHECK-NEXT:     "col": 11,
// CHECK-NEXT:     "tokLen": 2
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "offset": 124,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 9
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "offset": 156,
// CHECK-NEXT:      "line": 7,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 1
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "NS",
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "FunctionDecl",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "offset": 144,
// CHECK-NEXT:       "line": 6,
// CHECK-NEXT:       "col": 6,
// CHECK-NEXT:       "tokLen": 8
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 139,
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "tokLen": 4
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 153,
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "Function",
// CHECK-NEXT:      "mangledName": "_ZN4Test2NS8FunctionEv",
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void ()"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "FunctionDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "offset": 167,
// CHECK-NEXT:     "line": 8,
// CHECK-NEXT:     "col": 10,
// CHECK-NEXT:     "tokLen": 8
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "offset": 158,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 4
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "offset": 179,
// CHECK-NEXT:      "col": 22,
// CHECK-NEXT:      "tokLen": 1
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "parentDeclContextId": "0x{{.*}}",
// CHECK-NEXT:    "previousDecl": "0x{{.*}}",
// CHECK-NEXT:    "name": "Function",
// CHECK-NEXT:    "mangledName": "_ZN4Test2NS8FunctionEv",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 178,
// CHECK-NEXT:        "col": 21,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 179,
// CHECK-NEXT:        "col": 22,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CXXRecordDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "offset": 189,
// CHECK-NEXT:     "line": 10,
// CHECK-NEXT:     "col": 8,
// CHECK-NEXT:     "tokLen": 1
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "offset": 182,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 6
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "offset": 210,
// CHECK-NEXT:      "line": 12,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 1
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "name": "S",
// CHECK-NEXT:    "tagUsed": "struct",
// CHECK-NEXT:    "completeDefinition": true,
// CHECK-NEXT:    "definitionData": {
// CHECK-NEXT:     "canConstDefaultInit": true,
// CHECK-NEXT:     "canPassInRegisters": true,
// CHECK-NEXT:     "copyAssign": {
// CHECK-NEXT:      "hasConstParam": true,
// CHECK-NEXT:      "implicitHasConstParam": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "copyCtor": {
// CHECK-NEXT:      "hasConstParam": true,
// CHECK-NEXT:      "implicitHasConstParam": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "simple": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "defaultCtor": {
// CHECK-NEXT:      "defaultedIsConstexpr": true,
// CHECK-NEXT:      "exists": true,
// CHECK-NEXT:      "isConstexpr": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "dtor": {
// CHECK-NEXT:      "irrelevant": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "simple": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "hasConstexprNonCopyMoveConstructor": true,
// CHECK-NEXT:     "isAggregate": true,
// CHECK-NEXT:     "isEmpty": true,
// CHECK-NEXT:     "isLiteral": true,
// CHECK-NEXT:     "isPOD": true,
// CHECK-NEXT:     "isStandardLayout": true,
// CHECK-NEXT:     "isTrivial": true,
// CHECK-NEXT:     "isTriviallyCopyable": true,
// CHECK-NEXT:     "moveAssign": {
// CHECK-NEXT:      "exists": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "simple": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     },
// CHECK-NEXT:     "moveCtor": {
// CHECK-NEXT:      "exists": true,
// CHECK-NEXT:      "needsImplicit": true,
// CHECK-NEXT:      "simple": true,
// CHECK-NEXT:      "trivial": true
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXRecordDecl",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "offset": 189,
// CHECK-NEXT:       "line": 10,
// CHECK-NEXT:       "col": 8,
// CHECK-NEXT:       "tokLen": 1
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 182,
// CHECK-NEXT:        "col": 1,
// CHECK-NEXT:        "tokLen": 6
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 189,
// CHECK-NEXT:        "col": 8,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "isImplicit": true,
// CHECK-NEXT:      "name": "S",
// CHECK-NEXT:      "tagUsed": "struct"
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CXXMethodDecl",
// CHECK-NEXT:      "loc": {
// CHECK-NEXT:       "offset": 200,
// CHECK-NEXT:       "line": 11,
// CHECK-NEXT:       "col": 8,
// CHECK-NEXT:       "tokLen": 6
// CHECK-NEXT:      },
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 195,
// CHECK-NEXT:        "col": 3,
// CHECK-NEXT:        "tokLen": 4
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 207,
// CHECK-NEXT:        "col": 15,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      },
// CHECK-NEXT:      "name": "Method",
// CHECK-NEXT:      "mangledName": "_ZN4Test1S6MethodEv",
// CHECK-NEXT:      "type": {
// CHECK-NEXT:       "qualType": "void ()"
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   },
// CHECK-NEXT:   {
// CHECK-NEXT:    "id": "0x{{.*}}",
// CHECK-NEXT:    "kind": "CXXMethodDecl",
// CHECK-NEXT:    "loc": {
// CHECK-NEXT:     "offset": 221,
// CHECK-NEXT:     "line": 13,
// CHECK-NEXT:     "col": 9,
// CHECK-NEXT:     "tokLen": 6
// CHECK-NEXT:    },
// CHECK-NEXT:    "range": {
// CHECK-NEXT:     "begin": {
// CHECK-NEXT:      "offset": 213,
// CHECK-NEXT:      "col": 1,
// CHECK-NEXT:      "tokLen": 4
// CHECK-NEXT:     },
// CHECK-NEXT:     "end": {
// CHECK-NEXT:      "offset": 231,
// CHECK-NEXT:      "col": 19,
// CHECK-NEXT:      "tokLen": 1
// CHECK-NEXT:     }
// CHECK-NEXT:    },
// CHECK-NEXT:    "parentDeclContextId": "0x{{.*}}",
// CHECK-NEXT:    "previousDecl": "0x{{.*}}",
// CHECK-NEXT:    "name": "Method",
// CHECK-NEXT:    "mangledName": "_ZN4Test1S6MethodEv",
// CHECK-NEXT:    "type": {
// CHECK-NEXT:     "qualType": "void ()"
// CHECK-NEXT:    },
// CHECK-NEXT:    "inner": [
// CHECK-NEXT:     {
// CHECK-NEXT:      "id": "0x{{.*}}",
// CHECK-NEXT:      "kind": "CompoundStmt",
// CHECK-NEXT:      "range": {
// CHECK-NEXT:       "begin": {
// CHECK-NEXT:        "offset": 230,
// CHECK-NEXT:        "col": 18,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       },
// CHECK-NEXT:       "end": {
// CHECK-NEXT:        "offset": 231,
// CHECK-NEXT:        "col": 19,
// CHECK-NEXT:        "tokLen": 1
// CHECK-NEXT:       }
// CHECK-NEXT:      }
// CHECK-NEXT:     }
// CHECK-NEXT:    ]
// CHECK-NEXT:   }
// CHECK-NEXT:  ]
// CHECK-NEXT: }
