; ModuleID = "main"
target triple = "x86_64-pc-windows-msvc"
target datalayout = ""

declare i32 @"strlen"(i8* %".1")

declare i8* @"malloc"(i32 %".1")

declare i8* @"strcpy"(i8* %".1", i8* %".2")

declare i8* @"strcat"(i8* %".1", i8* %".2")

define i32 @"main"()
{
entry:
  %"a" = alloca i32
  store i32 10, i32* %"a"
  %"b" = alloca i32
  store i32 20, i32* %"b"
  %"c" = alloca i32
  store i32 30, i32* %"c"
  %"load_a" = load i32, i32* %"a"
  %"load_b" = load i32, i32* %"b"
  %"multmp" = mul i32 %"load_a", %"load_b"
  %"load_c" = load i32, i32* %"c"
  %"addtmp" = add i32 %"multmp", %"load_c"
  %"str_ptr_1" = getelementptr [3 x i8], [3 x i8]* @"str_1", i32 0, i32 0
  %"calltmp_0" = call i32 (i8*, ...) @"printf"(i8* %"str_ptr_1", i32 %"addtmp")
  %"str_ptr_2" = getelementptr [2 x i8], [2 x i8]* @"str_2", i32 0, i32 0
  %"calltmp_newline" = call i32 (i8*, ...) @"printf"(i8* %"str_ptr_2")
  ret i32 0
}

declare i32 @"printf"(i8* %".1", ...)

@"str_1" = internal constant [3 x i8] c"%d\00"
@"str_2" = internal constant [2 x i8] c"\0a\00"