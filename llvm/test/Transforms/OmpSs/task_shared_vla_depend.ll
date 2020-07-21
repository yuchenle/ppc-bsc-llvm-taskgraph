; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt %s -ompss-2 -S | FileCheck %s
; ModuleID = 'task_shared_vla_depend.c'
source_filename = "task_shared_vla_depend.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct._depend_unpack_t = type { i32*, i64, i64, i64, i64, i64, i64, i64, i64, i64 }

; void foo(int x, int y, int z) {
;     int vla[x + 1][y + 2][z + 3];
;     #pragma oss task in(vla)
;     {
;         int size = sizeof(vla);
;     }
; }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @foo(i32 %x, i32 %y, i32 %z) #0 !dbg !6 {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X_ADDR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[Y_ADDR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[Z_ADDR:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[SAVED_STACK:%.*]] = alloca i8*, align 8
; CHECK-NEXT:    [[__VLA_EXPR0:%.*]] = alloca i64, align 8
; CHECK-NEXT:    [[__VLA_EXPR1:%.*]] = alloca i64, align 8
; CHECK-NEXT:    [[__VLA_EXPR2:%.*]] = alloca i64, align 8
; CHECK-NEXT:    store i32 [[X:%.*]], i32* [[X_ADDR]], align 4
; CHECK-NEXT:    store i32 [[Y:%.*]], i32* [[Y_ADDR]], align 4
; CHECK-NEXT:    store i32 [[Z:%.*]], i32* [[Z_ADDR]], align 4
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, i32* [[X_ADDR]], align 4, !dbg !8
; CHECK-NEXT:    [[ADD:%.*]] = add nsw i32 [[TMP0]], 1, !dbg !9
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[ADD]] to i64, !dbg !10
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, i32* [[Y_ADDR]], align 4, !dbg !11
; CHECK-NEXT:    [[ADD1:%.*]] = add nsw i32 [[TMP2]], 2, !dbg !12
; CHECK-NEXT:    [[TMP3:%.*]] = zext i32 [[ADD1]] to i64, !dbg !10
; CHECK-NEXT:    [[TMP4:%.*]] = load i32, i32* [[Z_ADDR]], align 4, !dbg !13
; CHECK-NEXT:    [[ADD2:%.*]] = add nsw i32 [[TMP4]], 3, !dbg !14
; CHECK-NEXT:    [[TMP5:%.*]] = zext i32 [[ADD2]] to i64, !dbg !10
; CHECK-NEXT:    [[TMP6:%.*]] = call i8* @llvm.stacksave(), !dbg !10
; CHECK-NEXT:    store i8* [[TMP6]], i8** [[SAVED_STACK]], align 8, !dbg !10
; CHECK-NEXT:    [[TMP7:%.*]] = mul nuw i64 [[TMP1]], [[TMP3]], !dbg !10
; CHECK-NEXT:    [[TMP8:%.*]] = mul nuw i64 [[TMP7]], [[TMP5]], !dbg !10
; CHECK-NEXT:    [[VLA:%.*]] = alloca i32, i64 [[TMP8]], align 16, !dbg !10
; CHECK-NEXT:    store i64 [[TMP1]], i64* [[__VLA_EXPR0]], align 8, !dbg !10
; CHECK-NEXT:    store i64 [[TMP3]], i64* [[__VLA_EXPR1]], align 8, !dbg !10
; CHECK-NEXT:    store i64 [[TMP5]], i64* [[__VLA_EXPR2]], align 8, !dbg !10
; CHECK-NEXT:    br label [[FINAL_COND:%.*]], !dbg !15
; CHECK:       codeRepl:
; CHECK-NEXT:    [[TMP9:%.*]] = alloca %nanos6_task_args_foo0*, align 8, !dbg !15
; CHECK-NEXT:    [[TMP10:%.*]] = bitcast %nanos6_task_args_foo0** [[TMP9]] to i8**, !dbg !15
; CHECK-NEXT:    [[TMP11:%.*]] = alloca i8*, align 8, !dbg !15
; CHECK-NEXT:    [[TMP12:%.*]] = mul nuw i64 4, [[TMP1]], !dbg !15
; CHECK-NEXT:    [[TMP13:%.*]] = mul nuw i64 [[TMP12]], [[TMP3]], !dbg !15
; CHECK-NEXT:    [[TMP14:%.*]] = mul nuw i64 [[TMP13]], [[TMP5]], !dbg !15
; CHECK-NEXT:    [[TMP15:%.*]] = add nuw i64 0, [[TMP14]], !dbg !15
; CHECK-NEXT:    [[TMP16:%.*]] = add nuw i64 32, [[TMP15]], !dbg !15
; CHECK-NEXT:    call void @nanos6_create_task(%nanos6_task_info_t* @task_info_var_foo0, %nanos6_task_invocation_info_t* @task_invocation_info_foo0, i64 [[TMP16]], i8** [[TMP10]], i8** [[TMP11]], i64 0, i64 1), !dbg !15
; CHECK-NEXT:    [[TMP17:%.*]] = load %nanos6_task_args_foo0*, %nanos6_task_args_foo0** [[TMP9]], align 8, !dbg !15
; CHECK-NEXT:    [[TMP18:%.*]] = bitcast %nanos6_task_args_foo0* [[TMP17]] to i8*, !dbg !15
; CHECK-NEXT:    [[ARGS_END:%.*]] = getelementptr i8, i8* [[TMP18]], i64 32, !dbg !15
; CHECK-NEXT:    [[GEP_VLA:%.*]] = getelementptr [[NANOS6_TASK_ARGS_FOO0:%.*]], %nanos6_task_args_foo0* [[TMP17]], i32 0, i32 0, !dbg !15
; CHECK-NEXT:    [[TMP19:%.*]] = bitcast i32** [[GEP_VLA]] to i8**, !dbg !15
; CHECK-NEXT:    store i8* [[ARGS_END]], i8** [[TMP19]], align 4, !dbg !15
; CHECK-NEXT:    [[TMP20:%.*]] = mul nuw i64 4, [[TMP1]], !dbg !15
; CHECK-NEXT:    [[TMP21:%.*]] = mul nuw i64 [[TMP20]], [[TMP3]], !dbg !15
; CHECK-NEXT:    [[TMP22:%.*]] = mul nuw i64 [[TMP21]], [[TMP5]], !dbg !15
; CHECK-NEXT:    [[TMP23:%.*]] = getelementptr i8, i8* [[ARGS_END]], i64 [[TMP22]], !dbg !15
; CHECK-NEXT:    [[GEP_VLA1:%.*]] = getelementptr [[NANOS6_TASK_ARGS_FOO0]], %nanos6_task_args_foo0* [[TMP17]], i32 0, i32 0, !dbg !15
; CHECK-NEXT:    store i32* [[VLA]], i32** [[GEP_VLA1]], align 8, !dbg !15
; CHECK-NEXT:    [[CAPT_GEP_:%.*]] = getelementptr [[NANOS6_TASK_ARGS_FOO0]], %nanos6_task_args_foo0* [[TMP17]], i32 0, i32 1, !dbg !15
; CHECK-NEXT:    store i64 [[TMP1]], i64* [[CAPT_GEP_]], align 8, !dbg !15
; CHECK-NEXT:    [[CAPT_GEP_2:%.*]] = getelementptr [[NANOS6_TASK_ARGS_FOO0]], %nanos6_task_args_foo0* [[TMP17]], i32 0, i32 2, !dbg !15
; CHECK-NEXT:    store i64 [[TMP3]], i64* [[CAPT_GEP_2]], align 8, !dbg !15
; CHECK-NEXT:    [[CAPT_GEP_3:%.*]] = getelementptr [[NANOS6_TASK_ARGS_FOO0]], %nanos6_task_args_foo0* [[TMP17]], i32 0, i32 3, !dbg !15
; CHECK-NEXT:    store i64 [[TMP5]], i64* [[CAPT_GEP_3]], align 8, !dbg !15
; CHECK-NEXT:    [[TMP24:%.*]] = load i8*, i8** [[TMP11]], align 8, !dbg !15
; CHECK-NEXT:    call void @nanos6_submit_task(i8* [[TMP24]]), !dbg !15
; CHECK-NEXT:    br label [[FINAL_END:%.*]], !dbg !15
; CHECK:       final.end:
; CHECK-NEXT:    [[TMP25:%.*]] = load i8*, i8** [[SAVED_STACK]], align 8, !dbg !16
; CHECK-NEXT:    call void @llvm.stackrestore(i8* [[TMP25]]), !dbg !16
; CHECK-NEXT:    ret void, !dbg !16
; CHECK:       final.then:
; CHECK-NEXT:    [[SIZE_CLONE:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[TMP26:%.*]] = mul nuw i64 [[TMP1]], [[TMP3]], !dbg !17
; CHECK-NEXT:    [[TMP27:%.*]] = mul nuw i64 [[TMP26]], [[TMP5]], !dbg !17
; CHECK-NEXT:    [[TMP28:%.*]] = mul nuw i64 4, [[TMP27]], !dbg !17
; CHECK-NEXT:    [[CONV_CLONE:%.*]] = trunc i64 [[TMP28]] to i32, !dbg !17
; CHECK-NEXT:    store i32 [[CONV_CLONE]], i32* [[SIZE_CLONE]], align 4, !dbg !18
; CHECK-NEXT:    br label [[FINAL_END]], !dbg !16
; CHECK:       final.cond:
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @nanos6_in_final(), !dbg !15
; CHECK-NEXT:    [[TMP30:%.*]] = icmp ne i32 [[TMP29]], 0, !dbg !15
; CHECK-NEXT:    br i1 [[TMP30]], label [[FINAL_THEN:%.*]], label [[CODEREPL:%.*]], !dbg !15
;
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %z.addr = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  %__vla_expr1 = alloca i64, align 8
  %__vla_expr2 = alloca i64, align 8
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  store i32 %z, i32* %z.addr, align 4
  %0 = load i32, i32* %x.addr, align 4, !dbg !8
  %add = add nsw i32 %0, 1, !dbg !9
  %1 = zext i32 %add to i64, !dbg !10
  %2 = load i32, i32* %y.addr, align 4, !dbg !11
  %add1 = add nsw i32 %2, 2, !dbg !12
  %3 = zext i32 %add1 to i64, !dbg !10
  %4 = load i32, i32* %z.addr, align 4, !dbg !13
  %add2 = add nsw i32 %4, 3, !dbg !14
  %5 = zext i32 %add2 to i64, !dbg !10
  %6 = call i8* @llvm.stacksave(), !dbg !10
  store i8* %6, i8** %saved_stack, align 8, !dbg !10
  %7 = mul nuw i64 %1, %3, !dbg !10
  %8 = mul nuw i64 %7, %5, !dbg !10
  %vla = alloca i32, i64 %8, align 16, !dbg !10
  store i64 %1, i64* %__vla_expr0, align 8, !dbg !10
  store i64 %3, i64* %__vla_expr1, align 8, !dbg !10
  store i64 %5, i64* %__vla_expr2, align 8, !dbg !10
  %9 = call token @llvm.directive.region.entry() [ "DIR.OSS"([5 x i8] c"TASK\00"), "QUAL.OSS.SHARED"(i32* %vla), "QUAL.OSS.VLA.DIMS"(i32* %vla, i64 %1, i64 %3, i64 %5), "QUAL.OSS.CAPTURED"(i64 %1, i64 %3, i64 %5), "QUAL.OSS.DEP.IN"(i32* %vla, %struct._depend_unpack_t (i32*, i64, i64, i64)* @compute_dep, i32* %vla, i64 %1, i64 %3, i64 %5) ], !dbg !15
  %size = alloca i32, align 4
  %10 = mul nuw i64 %1, %3, !dbg !16
  %11 = mul nuw i64 %10, %5, !dbg !16
  %12 = mul nuw i64 4, %11, !dbg !16
  %conv = trunc i64 %12 to i32, !dbg !16
  store i32 %conv, i32* %size, align 4, !dbg !17
  call void @llvm.directive.region.exit(token %9), !dbg !18
  %13 = load i8*, i8** %saved_stack, align 8, !dbg !19
  call void @llvm.stackrestore(i8* %13), !dbg !19
  ret void, !dbg !19
}

; Function Attrs: nounwind
declare i8* @llvm.stacksave() #1

; Function Attrs: nounwind
declare token @llvm.directive.region.entry() #1

; Function Attrs: nounwind
declare void @llvm.directive.region.exit(token) #1

define internal %struct._depend_unpack_t @compute_dep(i32* %vla, i64 %0, i64 %1, i64 %2) {
; CHECK-LABEL: @compute_dep(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[RETURN_VAL:%.*]] = alloca [[STRUCT__DEPEND_UNPACK_T:%.*]], align 8
; CHECK-NEXT:    [[TMP3:%.*]] = mul i64 [[TMP2:%.*]], 4
; CHECK-NEXT:    [[TMP4:%.*]] = mul i64 [[TMP2]], 4
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 0
; CHECK-NEXT:    store i32* [[VLA:%.*]], i32** [[TMP5]], align 8
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 1
; CHECK-NEXT:    store i64 [[TMP3]], i64* [[TMP6]], align 8
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 2
; CHECK-NEXT:    store i64 0, i64* [[TMP7]], align 8
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 3
; CHECK-NEXT:    store i64 [[TMP4]], i64* [[TMP8]], align 8
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 4
; CHECK-NEXT:    store i64 [[TMP1:%.*]], i64* [[TMP9]], align 8
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 5
; CHECK-NEXT:    store i64 0, i64* [[TMP10]], align 8
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 6
; CHECK-NEXT:    store i64 [[TMP1]], i64* [[TMP11]], align 8
; CHECK-NEXT:    [[TMP12:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 7
; CHECK-NEXT:    store i64 [[TMP0:%.*]], i64* [[TMP12]], align 8
; CHECK-NEXT:    [[TMP13:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 8
; CHECK-NEXT:    store i64 0, i64* [[TMP13]], align 8
; CHECK-NEXT:    [[TMP14:%.*]] = getelementptr inbounds [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], i32 0, i32 9
; CHECK-NEXT:    store i64 [[TMP0]], i64* [[TMP14]], align 8
; CHECK-NEXT:    [[TMP15:%.*]] = load [[STRUCT__DEPEND_UNPACK_T]], %struct._depend_unpack_t* [[RETURN_VAL]], align 8
; CHECK-NEXT:    ret [[STRUCT__DEPEND_UNPACK_T]] %15
;
entry:
  %return.val = alloca %struct._depend_unpack_t, align 8
  %3 = mul i64 %2, 4
  %4 = mul i64 %2, 4
  %5 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 0
  store i32* %vla, i32** %5, align 8
  %6 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 1
  store i64 %3, i64* %6, align 8
  %7 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 2
  store i64 0, i64* %7, align 8
  %8 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 3
  store i64 %4, i64* %8, align 8
  %9 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 4
  store i64 %1, i64* %9, align 8
  %10 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 5
  store i64 0, i64* %10, align 8
  %11 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 6
  store i64 %1, i64* %11, align 8
  %12 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 7
  store i64 %0, i64* %12, align 8
  %13 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 8
  store i64 0, i64* %13, align 8
  %14 = getelementptr inbounds %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, i32 0, i32 9
  store i64 %0, i64* %14, align 8
  %15 = load %struct._depend_unpack_t, %struct._depend_unpack_t* %return.val, align 8
  ret %struct._depend_unpack_t %15
}

; CHECK: define internal void @nanos6_unpacked_task_region_foo0(i32* %vla, i64 %0, i64 %1, i64 %2, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) !dbg !19 {
; CHECK: newFuncRoot:
; CHECK-NEXT:   br label %3, !dbg !20
; CHECK: .exitStub:                                        ; preds = %3
; CHECK-NEXT:   ret void
; CHECK: 3:                                                ; preds = %newFuncRoot
; CHECK-NEXT:   %size = alloca i32, align 4
; CHECK-NEXT:   %4 = mul nuw i64 %0, %1, !dbg !20
; CHECK-NEXT:   %5 = mul nuw i64 %4, %2, !dbg !20
; CHECK-NEXT:   %6 = mul nuw i64 4, %5, !dbg !20
; CHECK-NEXT:   %conv = trunc i64 %6 to i32, !dbg !20
; CHECK-NEXT:   store i32 %conv, i32* %size, align 4, !dbg !21
; CHECK-NEXT:   br label %.exitStub, !dbg !22
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_task_region_foo0(%nanos6_task_args_foo0* %task_args, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %capt_gep1 = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep1 = load i64, i64* %capt_gep1
; CHECK-NEXT:   %capt_gep2 = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 3
; CHECK-NEXT:   %load_capt_gep2 = load i64, i64* %capt_gep2
; CHECK-NEXT:   call void @nanos6_unpacked_task_region_foo0(i32* %load_gep_vla, i64 %load_capt_gep, i64 %load_capt_gep1, i64 %load_capt_gep2, i8* %device_env, %nanos6_address_translation_entry_t* %address_translation_table)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_unpacked_deps_foo0(i32* %vla, i64 %0, i64 %1, i64 %2, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %3 = call %struct._depend_unpack_t @compute_dep(i32* %vla, i64 %0, i64 %1, i64 %2)
; CHECK-NEXT:   %4 = call %struct._depend_unpack_t @compute_dep(i32* %vla, i64 %0, i64 %1, i64 %2)
; CHECK-NEXT:   %5 = extractvalue %struct._depend_unpack_t %3, 0
; CHECK-NEXT:   %6 = bitcast i32* %5 to i8*
; CHECK-NEXT:   %7 = extractvalue %struct._depend_unpack_t %3, 1
; CHECK-NEXT:   %8 = extractvalue %struct._depend_unpack_t %3, 2
; CHECK-NEXT:   %9 = extractvalue %struct._depend_unpack_t %4, 3
; CHECK-NEXT:   %10 = extractvalue %struct._depend_unpack_t %3, 4
; CHECK-NEXT:   %11 = extractvalue %struct._depend_unpack_t %3, 5
; CHECK-NEXT:   %12 = extractvalue %struct._depend_unpack_t %4, 6
; CHECK-NEXT:   %13 = extractvalue %struct._depend_unpack_t %3, 7
; CHECK-NEXT:   %14 = extractvalue %struct._depend_unpack_t %3, 8
; CHECK-NEXT:   %15 = extractvalue %struct._depend_unpack_t %4, 9
; CHECK-NEXT:   call void @nanos6_register_region_read_depinfo3(i8* %handler, i32 0, i8* null, i8* %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13, i64 %14, i64 %15)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define internal void @nanos6_ol_deps_foo0(%nanos6_task_args_foo0* %task_args, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %gep_vla = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 0
; CHECK-NEXT:   %load_gep_vla = load i32*, i32** %gep_vla
; CHECK-NEXT:   %capt_gep = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 1
; CHECK-NEXT:   %load_capt_gep = load i64, i64* %capt_gep
; CHECK-NEXT:   %capt_gep1 = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 2
; CHECK-NEXT:   %load_capt_gep1 = load i64, i64* %capt_gep1
; CHECK-NEXT:   %capt_gep2 = getelementptr %nanos6_task_args_foo0, %nanos6_task_args_foo0* %task_args, i32 0, i32 3
; CHECK-NEXT:   %load_capt_gep2 = load i64, i64* %capt_gep2
; CHECK-NEXT:   call void @nanos6_unpacked_deps_foo0(i32* %load_gep_vla, i64 %load_capt_gep, i64 %load_capt_gep1, i64 %load_capt_gep2, %nanos6_loop_bounds_t* %loop_bounds, i8* %handler)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; Function Attrs: nounwind
declare void @llvm.stackrestore(i8*) #1

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "human", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "task_shared_vla_depend.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!""}
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 2, column: 13, scope: !6)
!9 = !DILocation(line: 2, column: 15, scope: !6)
!10 = !DILocation(line: 2, column: 5, scope: !6)
!11 = !DILocation(line: 2, column: 20, scope: !6)
!12 = !DILocation(line: 2, column: 22, scope: !6)
!13 = !DILocation(line: 2, column: 27, scope: !6)
!14 = !DILocation(line: 2, column: 29, scope: !6)
!15 = !DILocation(line: 3, column: 13, scope: !6)
!16 = !DILocation(line: 5, column: 20, scope: !6)
!17 = !DILocation(line: 5, column: 13, scope: !6)
!18 = !DILocation(line: 6, column: 5, scope: !6)
!19 = !DILocation(line: 7, column: 1, scope: !6)
