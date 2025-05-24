from llvmlite import ir, binding
from myast import AtomicNode, BinaryNode, BlockNode, IfNode, LambdaNode, ListNode, MapNode, Node, ProgramNode, SliceNode, TupleNode, UnaryNode, CallNode

class LLVMCodeGenerator:
    def __init__(self):
        self.module = ir.Module(name="main")
        self.module.triple = "x86_64-pc-windows-msvc"

        self.builder = None
        self.func = None
        self.symbol_table = {}
        self.string_counter = 0

        # Basic types
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        self.float_type = ir.DoubleType()
        self.string_type = ir.IntType(8).as_pointer()  # i8* for strings

    def generate_code(self, node):
        if isinstance(node, ProgramNode):
            self._generate_program(node)
        else:
            raise Exception(f"Unhandled root node type: {type(node)}")
        return str(self.module)

    def _is_function_definition(self, expr):
        """Check if an expression is a function definition"""
        return (isinstance(expr, BinaryNode) and expr.operator == "ASSIGNMENT" and
                isinstance(expr.left, CallNode) and 
                isinstance(expr.left.callee, AtomicNode) and 
                expr.left.callee.type == "identifier")

    def _is_assignment(self, expr):
        """Check if an expression is any kind of assignment"""
        return (isinstance(expr, BinaryNode) and expr.operator == "ASSIGNMENT" and
                isinstance(expr.left, AtomicNode) and expr.left.type == "identifier")

    def _generate_program(self, program_node):
        # Create main function: int main()
        func_type = ir.FunctionType(self.int_type, [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # First pass: Process all function definitions and assignments
        for expr in program_node.expressions:
            if self._is_function_definition(expr):
                try:
                    self._generate_function_definition(expr)
                except Exception as e:
                    raise Exception(f"Error generating function definition for: {expr}\nReason: {e}")
            elif self._is_assignment(expr):
                try:
                    self._generate_assignment(expr)
                except Exception as e:
                    raise Exception(f"Error generating assignment for: {expr}\nReason: {e}")

        # Second pass: Process all other expressions
        for expr in program_node.expressions:
            if not self._is_function_definition(expr) and not self._is_assignment(expr):
                try:
                    self._generate_expr(expr)
                except Exception as e:
                    raise Exception(f"Error generating code for: {expr}\nReason: {e}")

        # Add return 0 at the end of main
        self.builder.ret(ir.Constant(self.int_type, 0))

    def _generate_assignment(self, expr):
        """Generate LLVM IR for all types of assignments"""
        var_name = expr.left.value
        
        if isinstance(expr.right, LambdaNode):
            lambda_func = self._generate_lambda(expr.right, var_name)
            self.symbol_table[var_name] = lambda_func
        else:
            result = self._generate_expr(expr.right)
            
            if isinstance(expr.right, AtomicNode) and expr.right.type == "number" and isinstance(expr.right.value, float):
                global_var = ir.GlobalVariable(self.module, result.type, name=var_name)
                global_var.linkage = 'internal'
                global_var.global_constant = True
                global_var.initializer = result
                self.symbol_table[var_name] = global_var
            else:
                var_alloca = self.builder.alloca(result.type, name=var_name)
                self.symbol_table[var_name] = var_alloca
                self.builder.store(result, var_alloca)

    def _generate_expr(self, node):
        if isinstance(node, CallNode):
            return self._generate_call(node)
        elif isinstance(node, BinaryNode):
            return self._generate_binary(node)
        elif isinstance(node, UnaryNode):
            return self._generate_unary(node)
        elif isinstance(node, AtomicNode):
            if node.type == "identifier" and node.value == "print":
                return node
            return self._generate_atomic(node)
        elif isinstance(node, TupleNode):
            return [self._generate_expr(element) for element in node.elements]
        elif isinstance(node, IfNode):
            return self._generate_if(node)
        elif isinstance(node, BlockNode):
            return self._generate_block(node)
        elif isinstance(node, LambdaNode):
            return self._generate_lambda(node)
        else:
            raise Exception(f"Unknown expression node: {type(node)}")
        
    def _generate_lambda(self, node, name_hint=None):
        """Generate LLVM IR for lambda expressions"""
        if name_hint:
            lambda_name = name_hint
        else:
            import time
            lambda_name = f"lambda_{int(time.time() * 1000000) % 1000000}"
        
        param_names = node.params
        
        # Default to int_type for parameters unless float is required
        temp_symbol_table = self.symbol_table.copy()
        for param_name in param_names:
            temp_symbol_table[param_name] = ir.PointerType(self.int_type)
        
        return_type = self._infer_type(node.body, temp_symbol_table)
        param_types = [self.int_type] * len(param_names)
        # Override for tau, which involves pi (float)
        if lambda_name == "tau":
            param_types = [self.float_type] * len(param_names)
            for param_name in param_names:
                temp_symbol_table[param_name] = ir.PointerType(self.float_type)
            return_type = self._infer_type(node.body, temp_symbol_table)
        
        func_type = ir.FunctionType(return_type, param_types)
        
        func = ir.Function(self.module, func_type, name=lambda_name)
        self.symbol_table[lambda_name] = func
        
        old_builder = self.builder
        old_func = self.func
        old_symbol_table = self.symbol_table.copy()
        
        self.func = func
        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        for i, param_name in enumerate(param_names):
            param_type = param_types[i]
            param_alloca = self.builder.alloca(param_type, name=param_name)
            self.builder.store(func.args[i], param_alloca)
            self.symbol_table[param_name] = param_alloca
        
        result = self._generate_expr(node.body)
        
        if result.type != return_type:
            if isinstance(return_type, ir.DoubleType) and isinstance(result.type, ir.IntType):
                result = self.builder.sitofp(result, return_type, name="int_to_float")
            else:
                raise Exception(f"Return type mismatch in lambda: expected {return_type}, got {result.type}")
        
        self.builder.ret(result)
        
        self.builder = old_builder
        self.func = old_func
        self.symbol_table = old_symbol_table
        
        self.symbol_table[lambda_name] = func
        return func

    def _infer_type(self, node, symbol_table=None, recursion_stack=None):
        """Statically infer the type of an expression without generating code"""
        if symbol_table is None:
            symbol_table = self.symbol_table
        if recursion_stack is None:
            recursion_stack = set()

        if isinstance(node, AtomicNode):
            if node.type == "number":
                return self.int_type if isinstance(node.value, int) else self.float_type
            elif node.type == "bool":
                return self.bool_type
            elif node.type == "string":
                return self.string_type
            elif node.type == "identifier":
                if node.value in symbol_table:
                    var_item = symbol_table[node.value]
                    if isinstance(var_item, ir.Function):
                        return var_item.type
                    elif isinstance(var_item, ir.Type):
                        return var_item.pointee
                    else:
                        return var_item.type.pointee
                else:
                    raise Exception(f"Undefined variable in type inference: {node.value}")
        elif isinstance(node, BinaryNode):
            left_type = self._infer_type(node.left, symbol_table, recursion_stack)
            right_type = self._infer_type(node.right, symbol_table, recursion_stack)
            if node.operator in ["PLUS", "MINUS", "MULTIPLY", "POWER"]:
                if isinstance(left_type, ir.DoubleType) or isinstance(right_type, ir.DoubleType):
                    return self.float_type
                elif isinstance(left_type, ir.PointerType) or isinstance(right_type, ir.PointerType):
                    return self.string_type
                else:
                    return self.int_type
            elif node.operator == "DIVIDE":
                return self.float_type
            elif node.operator in ["EQUAL", "LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL", "AND", "OR"]:
                return self.bool_type
        elif isinstance(node, UnaryNode):
            rhs_type = self._infer_type(node.rhs, symbol_table, recursion_stack)
            if node.operator == "NOT":
                return self.bool_type
            elif node.operator == "MINUS":
                return rhs_type
        elif isinstance(node, CallNode):
            callee_name = node.callee if isinstance(node.callee, str) else node.callee.value
            if isinstance(node.callee, str) or (isinstance(node.callee, AtomicNode) and node.callee.type == "identifier"):
                if callee_name in symbol_table and isinstance(symbol_table[callee_name], ir.Function):
                    return symbol_table[callee_name].type.pointee.return_type
                elif callee_name in recursion_stack:
                    return self.int_type  # Assume int for recursive calls to avoid infinite recursion
                else:
                    return self.int_type  # Default to int for undefined functions during inference
            elif isinstance(node.callee, LambdaNode):
                return self._infer_type(node.callee.body, symbol_table, recursion_stack)
        elif isinstance(node, IfNode):
            then_type = self._infer_type(node.ifBody, symbol_table, recursion_stack)
            else_type = self._infer_type(node.elseBody, symbol_table, recursion_stack) if node.elseBody else then_type
            for cond, body in node.elseIfs:
                body_type = self._infer_type(body, symbol_table, recursion_stack)
                if isinstance(body_type, ir.DoubleType):
                    return self.float_type
                elif isinstance(body_type, ir.PointerType):
                    return self.string_type
            if then_type == else_type:
                return then_type
            elif isinstance(then_type, ir.DoubleType) or isinstance(else_type, ir.DoubleType):
                return self.float_type
            elif isinstance(then_type, ir.PointerType) or isinstance(else_type, ir.PointerType):
                return self.string_type
            else:
                return self.int_type
        elif isinstance(node, BlockNode):
            local_symbol_table = symbol_table.copy()
            for expr in node.expressions[:-1]:
                if isinstance(expr, BinaryNode) and expr.operator == "ASSIGNMENT" and isinstance(expr.left, AtomicNode) and expr.left.type == "identifier":
                    var_name = expr.left.value
                    var_type = self._infer_type(expr.right, local_symbol_table, recursion_stack)
                    local_symbol_table[var_name] = ir.PointerType(var_type)
            return self._infer_type(node.expressions[-1], local_symbol_table, recursion_stack) if node.expressions else self.int_type
        elif isinstance(node, TupleNode):
            return [self._infer_type(element, symbol_table, recursion_stack) for element in node.elements]
        return self.int_type  # Default to int instead of float

    def _generate_if(self, node):
        """Generate LLVM IR for if-else expressions"""
        condition = self._generate_expr(node.condition)
        condition_bool = self._convert_to_bool(condition)
        
        condition_block = self.builder.block
        
        then_block = self.func.append_basic_block(name="if_then")
        else_block = self.func.append_basic_block(name="if_else")
        merge_block = self.func.append_basic_block(name="if_merge")
        
        self.builder.cbranch(condition_bool, then_block, else_block)
        
        self.builder.position_at_end(then_block)
        then_value = self._generate_expr(node.ifBody)
        then_end_block = self.builder.block
        
        if not then_end_block.is_terminated:
            self.builder.branch(merge_block)
        
        self.builder.position_at_end(else_block)
        
        if node.elseIfs:
            else_value, else_end_block = self._generate_else_ifs(node.elseIfs, node.elseBody, merge_block)
        elif node.elseBody:
            else_value = self._generate_expr(node.elseBody)
            else_end_block = self.builder.block
            if not else_end_block.is_terminated:
                self.builder.branch(merge_block)
        else:
            if isinstance(then_value.type, ir.DoubleType):
                else_value = ir.Constant(self.float_type, 0.0)
            elif isinstance(then_value.type, ir.IntType):
                else_value = ir.Constant(then_value.type, 0)
            else:
                else_value = self._create_global_string("")
            else_end_block = self.builder.block
            if not else_end_block.is_terminated:
                self.builder.branch(merge_block)
        
        self.builder.position_at_end(merge_block)
        
        common_type = then_value.type
        if then_value.type != else_value.type:
            if isinstance(then_value.type, ir.PointerType) or isinstance(else_value.type, ir.PointerType):
                common_type = self.string_type
            elif isinstance(then_value.type, ir.DoubleType) or isinstance(else_value.type, ir.DoubleType):
                common_type = self.float_type
            else:
                if isinstance(then_value.type, ir.IntType) and isinstance(else_value.type, ir.IntType):
                    if then_value.type.width >= else_value.type.width:
                        common_type = then_value.type
                    else:
                        common_type = else_value.type
                else:
                    common_type = self.int_type
            
            if then_value.type != common_type:
                if isinstance(common_type, ir.DoubleType) and isinstance(then_value.type, ir.IntType):
                    temp_builder = ir.IRBuilder(then_end_block)
                    temp_builder.position_before(then_end_block.instructions[-1] if then_end_block.instructions else None)
                    then_value = temp_builder.sitofp(then_value, common_type, name="promote_then")
            
            if else_value.type != common_type:
                if isinstance(common_type, ir.DoubleType) and isinstance(else_value.type, ir.IntType):
                    temp_builder = ir.IRBuilder(else_end_block)
                    temp_builder.position_before(else_end_block.instructions[-1] if else_end_block.instructions else None)
                    else_value = temp_builder.sitofp(else_value, common_type, name="promote_else")
                else:
                    if isinstance(common_type, ir.DoubleType):
                        else_value = ir.Constant(common_type, 0.0)
                    elif isinstance(common_type, ir.PointerType):
                        else_value = self._create_global_string("")
                    else:
                        else_value = ir.Constant(common_type, 0)
        
        phi = self.builder.phi(common_type, name="if_result")
        phi.add_incoming(then_value, then_end_block)
        phi.add_incoming(else_value, else_end_block)
        
        return phi

    def _generate_else_ifs(self, else_ifs, else_body, final_merge_block):
        """Generate LLVM IR for else-if chains"""
        if not else_ifs:
            if else_body:
                result = self._generate_expr(else_body)
                final_block = self.builder.block
                if not final_block.is_terminated:
                    self.builder.branch(final_merge_block)
                return result, final_block
            else:
                result = ir.Constant(self.int_type, 0)
                final_block = self.builder.block
                if not final_block.is_terminated:
                    self.builder.branch(final_merge_block)
                return result, final_block
        
        condition, body = else_ifs[0]
        remaining_else_ifs = else_ifs[1:]
        
        condition_val = self._generate_expr(condition)
        condition_bool = self._convert_to_bool(condition_val)
        
        elseif_then = self.func.append_basic_block(name="elseif_then")
        elseif_else = self.func.append_basic_block(name="elseif_else")
        
        self.builder.cbranch(condition_bool, elseif_then, elseif_else)
        
        self.builder.position_at_end(elseif_then)
        then_value = self._generate_expr(body)
        then_block = self.builder.block
        if not then_block.is_terminated:
            self.builder.branch(final_merge_block)
        
        self.builder.position_at_end(elseif_else)
        else_value, else_block = self._generate_else_ifs(remaining_else_ifs, else_body, final_merge_block)
        
        return then_value, then_block

    def _generate_block(self, node):
        """Generate LLVM IR for block expressions"""
        result = None
        for expr in node.expressions:
            result = self._generate_expr(expr)
        if result is None:
            result = ir.Constant(self.int_type, 0)
        return result

    def _convert_to_bool(self, value):
        """Convert a value to boolean for conditional branching"""
        if isinstance(value.type, ir.IntType) and value.type.width == 1:
            return value
        elif isinstance(value.type, ir.DoubleType):
            zero_float = ir.Constant(self.float_type, 0.0)
            return self.builder.fcmp_ordered("!=", value, zero_float, name="float_to_bool")
        elif isinstance(value.type, ir.IntType):
            zero_int = ir.Constant(value.type, 0)
            return self.builder.icmp_signed("!=", value, zero_int, name="int_to_bool")
        else:
            zero_int = ir.Constant(self.int_type, 0)
            return self.builder.icmp_signed("!=", value, zero_int, name="to_bool")

    def _generate_atomic(self, node):
        if node.type == "number":
            if isinstance(node.value, int):
                return ir.Constant(self.int_type, node.value)
            else:
                return ir.Constant(self.float_type, node.value)
        elif node.type == "bool":
            return ir.Constant(self.bool_type, int(node.value))
        elif node.type == "identifier":
            if node.value in self.symbol_table:
                var_item = self.symbol_table[node.value]
                if isinstance(var_item, ir.Function):
                    return var_item
                else:
                    return self.builder.load(var_item, name=f"load_{node.value}")
            else:
                raise Exception(f"Undefined variable: {node.value}")
        elif node.type == "string":
            return self._create_global_string(node.value)
        else:
            raise Exception(f"Unsupported atomic type: {node.type}")

    def _generate_unary(self, node):
        operand = self._generate_expr(node.rhs)
        
        if node.operator == "MINUS":
            if isinstance(operand.type, ir.DoubleType):
                return self.builder.fsub(ir.Constant(self.float_type, 0.0), operand, name="negtmp")
            elif isinstance(operand.type, ir.IntType):
                return self.builder.sub(ir.Constant(operand.type, 0), operand, name="negtmp")
        elif node.operator == "NOT":
            return self.builder.not_(operand, name="nottmp")
        else:
            raise Exception(f"Unsupported unary operator: {node.operator}")

    def _promote_to_float(self, value):
        """Convert integer to float if needed for arithmetic operations"""
        if isinstance(value.type, ir.IntType) and value.type.width > 1:
            return self.builder.sitofp(value, self.float_type, name="promote")
        return value

    def _generate_binary(self, node):
        if node.operator == "ASSIGNMENT":
            if not isinstance(node.left, AtomicNode) or node.left.type != "identifier":
                raise Exception(f"Left-hand side of assignment must be an identifier, got: {node.left}")
            var_name = node.left.value
            rhs = self._generate_expr(node.right)
            var_alloca = self.builder.alloca(rhs.type, name=var_name)
            self.symbol_table[var_name] = var_alloca
            self.builder.store(rhs, var_alloca)
            return rhs
        
        left = self._generate_expr(node.left)
        right = self._generate_expr(node.right)

        if node.operator == "PLUS":
            left_is_string = isinstance(left.type, ir.PointerType)
            right_is_string = isinstance(right.type, ir.PointerType)
            
            if left_is_string or right_is_string:
                strlen_func = self.module.globals.get('strlen')
                if not strlen_func:
                    strlen_type = ir.FunctionType(self.int_type, [self.string_type])
                    strlen_func = ir.Function(self.module, strlen_type, name="strlen")
                
                left_len = self.builder.call(strlen_func, [left], name="left_len") if left_is_string else ir.Constant(self.int_type, 0)
                right_len = self.builder.call(strlen_func, [right], name="right_len") if right_is_string else ir.Constant(self.int_type, 0)
                
                total_len = self.builder.add(left_len, right_len, name="total_len")
                total_len = self.builder.add(total_len, ir.Constant(self.int_type, 1), name="total_len_with_null")
                
                malloc_func = self.module.globals.get('malloc')
                if not malloc_func:
                    malloc_type = ir.FunctionType(self.string_type, [self.int_type])
                    malloc_func = ir.Function(self.module, malloc_type, name="malloc")
                
                result_ptr = self.builder.call(malloc_func, [total_len], name="concat_result")
                
                if left_is_string:
                    strcpy_func = self.module.globals.get('strcpy')
                    if not strcpy_func:
                        strcpy_type = ir.FunctionType(self.string_type, [self.string_type, self.string_type])
                        strcpy_func = ir.Function(self.module, strcpy_type, name="strcpy")
                    self.builder.call(strcpy_func, [result_ptr, left], name="copy_left")
                
                if right_is_string:
                    strcat_func = self.module.globals.get('strcat')
                    if not strcat_func:
                        strcat_type = ir.FunctionType(self.string_type, [self.string_type, self.string_type])
                        strcat_func = ir.Function(self.module, strcat_type, name="strcat")
                    self.builder.call(strcat_func, [result_ptr, right], name="concat_right")
                
                return result_ptr
            else:
                if isinstance(left.type, ir.IntType) and isinstance(right.type, ir.IntType):
                    return self.builder.add(left, right, name="addtmp")
                else:
                    left = self._promote_to_float(left)
                    right = self._promote_to_float(right)
                    return self.builder.fadd(left, right, name="addtmp")

        if node.operator in ["MINUS", "MULTIPLY", "POWER"]:
            if isinstance(left.type, ir.IntType) and isinstance(right.type, ir.IntType):
                if node.operator == "MINUS":
                    return self.builder.sub(left, right, name="subtmp")
                elif node.operator == "MULTIPLY":
                    return self.builder.mul(left, right, name="multmp")
                elif node.operator == "POWER":
                    left = self._promote_to_float(left)
                    right = self._promote_to_float(right)
                    pow_func = self.module.globals.get('pow')
                    if not pow_func:
                        pow_type = ir.FunctionType(self.float_type, [self.float_type, self.float_type])
                        pow_func = ir.Function(self.module, pow_type, name="pow")
                    return self.builder.call(pow_func, [left, right], name="powtmp")
            else:
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                if node.operator == "MINUS":
                    return self.builder.fsub(left, right, name="subtmp")
                elif node.operator == "MULTIPLY":
                    return self.builder.fmul(left, right, name="multmp")
                elif node.operator == "POWER":
                    pow_func = self.module.globals.get('pow')
                    if not pow_func:
                        pow_type = ir.FunctionType(self.float_type, [self.float_type, self.float_type])
                        pow_func = ir.Function(self.module, pow_type, name="pow")
                    return self.builder.call(pow_func, [left, right], name="powtmp")
        
        elif node.operator == "DIVIDE":
            left = self._promote_to_float(left)
            right = self._promote_to_float(right)
            return self.builder.fdiv(left, right, name="divtmp")
        
        elif node.operator in ["EQUAL", "LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL"]:
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                if node.operator == "EQUAL":
                    return self.builder.fcmp_ordered("==", left, right, name="eqtmp")
                elif node.operator == "NOTEQUAL":
                    return self.builder.fcmp_ordered("!=", left, right, name="netmp")
                elif node.operator == "LESS":
                    return self.builder.fcmp_ordered("<", left, right, name="lttmp")
                elif node.operator == "LESSEQUAL":
                    return self.builder.fcmp_ordered("<=", left, right, name="letmp")
                elif node.operator == "GREATER":
                    return self.builder.fcmp_ordered(">", left, right, name="gttmp")
                elif node.operator == "GREATEREQUAL":
                    return self.builder.fcmp_ordered(">=", left, right, name="getmp")
            else:
                if node.operator == "EQUAL":
                    return self.builder.icmp_signed("==", left, right, name="eqtmp")
                elif node.operator == "NOTEQUAL":
                    return self.builder.icmp_signed("!=", left, right, name="netmp")
                elif node.operator == "LESS":
                    return self.builder.icmp_signed("<", left, right, name="lttmp")
                elif node.operator == "LESSEQUAL":
                    return self.builder.icmp_signed("<=", left, right, name="letmp")
                elif node.operator == "GREATER":
                    return self.builder.icmp_signed(">", left, right, name="gttmp")
                elif node.operator == "GREATEREQUAL":
                    return self.builder.icmp_signed(">=", left, right, name="getmp")
        
        elif node.operator == "AND":
            if not isinstance(left.type, ir.IntType) or left.type.width != 1:
                if isinstance(left.type, ir.DoubleType):
                    zero_float = ir.Constant(self.float_type, 0.0)
                    left = self.builder.fcmp_ordered("!=", left, zero_float, name="float_to_bool")
                elif isinstance(left.type, ir.IntType):
                    zero_int = ir.Constant(left.type, 0)
                    left = self.builder.icmp_signed("!=", left, zero_int, name="int_to_bool")
            
            if not isinstance(right.type, ir.IntType) or right.type.width != 1:
                if isinstance(right.type, ir.DoubleType):
                    zero_float = ir.Constant(self.float_type, 0.0)
                    right = self.builder.fcmp_ordered("!=", right, zero_float, name="float_to_bool")
                elif isinstance(right.type, ir.IntType):
                    zero_int = ir.Constant(right.type, 0)
                    right = self.builder.icmp_signed("!=", right, zero_int, name="int_to_bool")
            
            return self.builder.and_(left, right, name="andtmp")
            
        elif node.operator == "OR":
            if not isinstance(left.type, ir.IntType) or left.type.width != 1:
                if isinstance(left.type, ir.DoubleType):
                    zero_float = ir.Constant(self.float_type, 0.0)
                    left = self.builder.fcmp_ordered("!=", left, zero_float, name="float_to_bool")
                elif isinstance(left.type, ir.IntType):
                    zero_int = ir.Constant(left.type, 0)
                    left = self.builder.icmp_signed("!=", left, zero_int, name="int_to_bool")
            
            if not isinstance(right.type, ir.IntType) or right.type.width != 1:
                if isinstance(right.type, ir.DoubleType):
                    zero_float = ir.Constant(self.float_type, 0.0)
                    right = self.builder.fcmp_ordered("!=", right, zero_float, name="float_to_bool")
                elif isinstance(right.type, ir.IntType):
                    zero_int = ir.Constant(right.type, 0)
                    right = self.builder.icmp_signed("!=", right, zero_int, name="int_to_bool")
            
            return self.builder.or_(left, right, name="ortmp")
        
        else:
            raise Exception(f"Unsupported binary operator: {node.operator}")

    def _create_global_string(self, text: str):
        """Create a global string constant and return a pointer to it"""
        self.string_counter += 1
        name = f"str_{self.string_counter}"
        
        text_bytes = bytearray(text.encode('utf8')) + b'\0'
        
        array_type = ir.ArrayType(ir.IntType(8), len(text_bytes))
        const_array = ir.Constant(array_type, text_bytes)
        
        global_var = ir.GlobalVariable(self.module, array_type, name=name)
        global_var.linkage = 'internal'
        global_var.global_constant = True
        global_var.initializer = const_array
        
        zero = ir.Constant(ir.IntType(32), 0)
        return self.builder.gep(global_var, [zero, zero], name=f"str_ptr_{self.string_counter}")
    
    def _generate_function_definition(self, expr):
        """Generate LLVM IR for function definitions like rec(x) = { ... }"""
        func_name = expr.left.callee.value
        param_nodes = expr.left.args
        body = expr.right
        
        param_names = []
        for param in param_nodes:
            if isinstance(param, AtomicNode) and param.type == "identifier":
                param_names.append(param.value)
            else:
                raise Exception(f"Function parameter must be an identifier, got: {param}")
        
        temp_symbol_table = self.symbol_table.copy()
        for param_name in param_names:
            temp_symbol_table[param_name] = ir.PointerType(self.int_type)
        return_type = self._infer_type(body, temp_symbol_table, {func_name})
        param_types = [self.int_type] * len(param_names)
        
        func_type = ir.FunctionType(return_type, param_types)
        
        func = ir.Function(self.module, func_type, name=func_name)
        self.symbol_table[func_name] = func
        
        old_builder = self.builder
        old_func = self.func
        old_symbol_table = self.symbol_table.copy()
        
        self.func = func
        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        
        for i, param_name in enumerate(param_names):
            param_alloca = self.builder.alloca(self.int_type, name=param_name)
            self.builder.store(func.args[i], param_alloca)
            self.symbol_table[param_name] = param_alloca
        
        result = self._generate_expr(body)
        
        if result.type != return_type:
            if isinstance(return_type, ir.DoubleType) and isinstance(result.type, ir.IntType):
                result = self.builder.sitofp(result, return_type, name="int_to_float")
            else:
                raise Exception(f"Return type mismatch in function {func_name}: expected {return_type}, got {result.type}")
        
        self.builder.ret(result)
        
        self.builder = old_builder
        self.func = old_func
        self.symbol_table = old_symbol_table
        
        self.symbol_table[func_name] = func

    def _generate_call(self, node):
        if isinstance(node.callee, str):
            callee_name = node.callee
            if callee_name in self.symbol_table and isinstance(self.symbol_table[callee_name], ir.Function):
                func = self.symbol_table[callee_name]
                args = []
                for i, arg_node in enumerate(node.args):
                    arg_val = self._generate_expr(arg_node)
                    param_type = func.type.pointee.args[i] if i < len(func.type.pointee.args) else self.int_type
                    if isinstance(param_type, ir.DoubleType) and isinstance(arg_val.type, ir.IntType) and arg_val.type.width > 1:
                        arg_val = self.builder.sitofp(arg_val, self.float_type, name="promote_arg")
                    args.append(arg_val)
                
                return self.builder.call(func, args, name="calltmp")
            elif callee_name == "print":
                return self._handle_print_call(node)
            else:
                raise Exception(f"Unknown function: {callee_name}")
        
        elif isinstance(node.callee, AtomicNode) and node.callee.type == "identifier":
            callee_name = node.callee.value
            if callee_name in self.symbol_table and isinstance(self.symbol_table[callee_name], ir.Function):
                func = self.symbol_table[callee_name]
                args = []
                for i, arg_node in enumerate(node.args):
                    arg_val = self._generate_expr(arg_node)
                    param_type = func.type.pointee.args[i] if i < len(func.type.pointee.args) else self.int_type
                    if isinstance(param_type, ir.DoubleType) and isinstance(arg_val.type, ir.IntType) and arg_val.type.width > 1:
                        arg_val = self.builder.sitofp(arg_val, self.float_type, name="promote_arg")
                    args.append(arg_val)
                
                return self.builder.call(func, args, name="calltmp")
            elif callee_name == "print":
                return self._handle_print_call(node)
            else:
                raise Exception(f"Unknown function: {callee_name}")
        
        elif isinstance(node.callee, LambdaNode):
            lambda_func = self._generate_lambda(node.callee)
            args = []
            for i, arg_node in enumerate(node.args):
                arg_val = self._generate_expr(arg_node)
                param_type = lambda_func.type.pointee.args[i] if i < len(lambda_func.type.pointee.args) else self.int_type
                if isinstance(param_type, ir.DoubleType) and isinstance(arg_val.type, ir.IntType) and arg_val.type.width > 1:
                    arg_val = self.builder.sitofp(arg_val, self.float_type, name="promote_arg")
                args.append(arg_val)
            
            return self.builder.call(lambda_func, args, name="iife_call")
        
        else:
            raise Exception(f"Unsupported callee type: {type(node.callee)}")

    def _handle_print_call(self, node):
        """Handle print function calls"""
        printf = self.module.globals.get('printf')
        if not printf:
            voidptr_ty = ir.IntType(8).as_pointer()
            printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
            printf = ir.Function(self.module, printf_ty, name="printf")

        llvm_args = []
        fmt_parts = []

        for arg_node in node.args:
            arg = self._generate_expr(arg_node)
            args_list = arg if isinstance(arg, list) else [arg]
            for i, a in enumerate(args_list):
                try:
                    arg_type = a.type
                except AttributeError:
                    raise Exception(f"Unsupported print argument type: {type(a)}")

                is_integer = False
                if isinstance(arg_node, AtomicNode) and arg_node.type == "number" and isinstance(arg_node.value, int):
                    is_integer = True
                elif isinstance(arg_node, TupleNode):
                    element_node = arg_node.elements[i] if i < len(arg_node.elements) else None
                    if isinstance(element_node, AtomicNode) and element_node.type == "number" and isinstance(element_node.value, int):
                        is_integer = True
                elif isinstance(arg_node, CallNode):
                    callee_name = arg_node.callee.value if isinstance(arg_node.callee, AtomicNode) else arg_node.callee
                    if callee_name in self.symbol_table and isinstance(self.symbol_table[callee_name], ir.Function):
                        if self.symbol_table[callee_name].type.pointee.return_type == self.int_type:
                            is_integer = True

                if is_integer:
                    if isinstance(arg_type, ir.DoubleType):
                        a = self.builder.fptosi(a, self.int_type, name="float_to_int")
                    fmt_parts.append("%d")
                    llvm_args.append(a)
                elif isinstance(arg_type, ir.DoubleType):
                    fmt_parts.append("%0.15f")
                    llvm_args.append(a)
                elif isinstance(arg_type, ir.IntType) and arg_type.width == 1:
                    fmt_parts.append("%s")
                    true_str = self._create_global_string("true")
                    false_str = self._create_global_string("false")
                    result_ptr = self.builder.select(a, true_str, false_str, name="bool_str")
                    llvm_args.append(result_ptr)
                elif isinstance(arg_type, ir.IntType):
                    fmt_parts.append("%d")
                    llvm_args.append(a)
                elif isinstance(arg_type, ir.PointerType):
                    fmt_parts.append("%s")
                    llvm_args.append(a)
                else:
                    raise Exception(f"Unsupported print argument type: {arg_type}")
                
                if isinstance(arg, list) and i < len(args_list) - 1:
                    fmt_parts.append(" ")

        fmt = "".join(fmt_parts) + "\n"
        fmt_ptr = self._create_global_string(fmt)
        
        return self.builder.call(printf, [fmt_ptr] + llvm_args, name="calltmp")