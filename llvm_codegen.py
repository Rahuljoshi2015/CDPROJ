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
        self.lambda_counter = 0  # Added for unique lambda names

        # Basic types
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        self.float_type = ir.DoubleType()
        self.string_type = ir.IntType(8).as_pointer()  # i8* for strings

    def _dprint(self, *args):
        """Debug print method"""
        print(*args)

    def generate_code(self, node):
        if isinstance(node, ProgramNode):
            self._generate_program(node)
        else:
            raise Exception(f"Unhandled root node type: {type(node)}")
        self._dprint("Generated LLVM IR:")
        self._dprint(str(self.module))
        return str(self.module)

    def _is_function_definition(self, expr):
        return (isinstance(expr, BinaryNode) and expr.operator == "ASSIGNMENT" and
                isinstance(expr.left, CallNode) and 
                isinstance(expr.left.callee, AtomicNode) and 
                expr.left.callee.type == "identifier")

    def _is_assignment(self, expr):
        return (isinstance(expr, BinaryNode) and expr.operator == "ASSIGNMENT" and
                isinstance(expr.left, AtomicNode) and expr.left.type == "identifier")

    def _generate_program(self, program_node):
        func_type = ir.FunctionType(self.int_type, [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        for expr in program_node.expressions:
            if self._is_function_definition(expr):
                try:
                    self._generate_function_definition(expr)
                except Exception as e:
                    raise Exception(f"Error generating function definition for: {expr}\nReason: {str(e)}")
            elif self._is_assignment(expr):
                try:
                    self._generate_assignment(expr)
                except Exception as e:
                    raise Exception(f"Error generating assignment for: {expr}\nReason: {str(e)}")
            else:
                try:
                    self._generate_expr(expr)
                except Exception as e:
                    raise Exception(f"Error generating code for: {expr}\nReason: {str(e)}")

        self.builder.ret(ir.Constant(self.int_type, 0))

    def _generate_assignment(self, expr):
        var_name = expr.left.value
        self._dprint(f"Starting assignment for: {var_name}, symbol_table keys: {list(self.symbol_table.keys())}")
        try:
            result = self._generate_expr(expr.right)
            self._dprint(f"Assignment: {var_name} = {result} (type: {result.type})")
            
            if var_name == "pi" and isinstance(result, ir.Constant) and isinstance(result.type, ir.DoubleType):
                # Create a global constant for pi
                global_var = ir.GlobalVariable(self.module, result.type, name=var_name)
                global_var.linkage = 'internal'
                global_var.global_constant = True
                global_var.initializer = result
                self.symbol_table[var_name] = global_var
            else:
                var_alloca = self.builder.alloca(result.type, name=var_name)
                self.symbol_table[var_name] = var_alloca
                self.builder.store(result, var_alloca)
            self._dprint(f"Completed assignment for: {var_name}, symbol_table keys: {list(self.symbol_table.keys())}")
        except Exception as e:
            raise Exception(f"Failed to generate assignment for {var_name} with expression {expr.right}: {str(e)}")

    def _generate_expr(self, node):
        self._dprint(f"Generating expr: {node}")
        if isinstance(node, CallNode):
            result = self._generate_call(node)
            self._dprint(f"Call result: {result} (type: {result.type})")
            return result
        elif isinstance(node, BinaryNode):
            result = self._generate_binary(node)
            self._dprint(f"Binary result: {result} (type: {result.type})")
            return result
        elif isinstance(node, UnaryNode):
            result = self._generate_unary(node)
            self._dprint(f"Unary result: {result} (type: {result.type})")
            return result
        elif isinstance(node, AtomicNode):
            if node.type == "identifier" and node.value == "print":
                return node
            result = self._generate_atomic(node)
            self._dprint(f"Atomic result: {result} (type: {result.type})")
            return result
        elif isinstance(node, TupleNode):
            result = [self._generate_expr(element) for element in node.elements]
            self._dprint(f"Tuple result: {result}")
            return result
        elif isinstance(node, IfNode):
            result = self._generate_if(node)
            self._dprint(f"If result: {result} (type: {result.type})")
            return result
        elif isinstance(node, BlockNode):
            result = self._generate_block(node)
            self._dprint(f"Block result: {result} (type: {result.type})")
            return result
        elif isinstance(node, LambdaNode):
            result = self._generate_lambda(node)
            self._dprint(f"Lambda result: {result} (type: {result.type})")
            return result
        else:
            raise Exception(f"Unknown expression node: {type(node)}")

    def _generate_lambda(self, node, name_hint=None):
        self._dprint(f"Generating lambda with params: {node.params}")
        if name_hint:
            lambda_name = name_hint
        else:
            self.lambda_counter += 1
            lambda_name = f"lambda_{self.lambda_counter}"
        
        param_names = node.params
        temp_symbol_table = self.symbol_table.copy()
        for param_name in param_names:
            temp_symbol_table[param_name] = ir.PointerType(self.int_type)
        
        return_type = self._infer_type(node.body, temp_symbol_table)
        param_types = [self.int_type] * len(param_names)
        
        func_type = ir.FunctionType(return_type, param_types)
        func = ir.Function(self.module, func_type, name=lambda_name)
        
        old_builder = self.builder
        old_func = self.func
        old_symbol_table = self.symbol_table
        
        self.func = func
        entry_block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(entry_block)
        self.symbol_table = temp_symbol_table
        
        self._dprint(f"Lambda {lambda_name} symbol_table keys before body: {list(self.symbol_table.keys())}")
        
        for i, param_name in enumerate(param_names):
            param_type = param_types[i]
            param_alloca = self.builder.alloca(param_type, name=param_name)
            self.symbol_table[param_name] = param_alloca
            self.builder.store(func.args[i], param_alloca)
        
        try:
            result = self._generate_expr(node.body)
            
            if result.type != return_type:
                if isinstance(return_type, ir.DoubleType) and isinstance(result.type, ir.IntType):
                    result = self.builder.sitofp(result, return_type, name="int_to_float")
                else:
                    raise Exception(f"Return type mismatch in lambda {lambda_name}: expected {return_type}, got {result.type}")
            
            self.builder.ret(result)
        except Exception as e:
            raise Exception(f"Failed to generate lambda body for {lambda_name}: {str(e)}")
        
        self.builder = old_builder
        self.func = old_func
        self.symbol_table = old_symbol_table
        
        self.symbol_table[lambda_name] = func
        self._dprint(f"Lambda {lambda_name} stored, symbol_table keys: {list(self.symbol_table.keys())}")
        return func

    def _infer_type(self, node, symbol_table=None, recursion_stack=None):
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
            if node.operator in ["PLUS", "MINUS", "MULTIPLY", "POWER", "DIVIDE"]:
                if isinstance(left_type, ir.DoubleType) or isinstance(right_type, ir.DoubleType) or node.operator == "DIVIDE":
                    return self.float_type
                elif isinstance(left_type, ir.PointerType) and isinstance(right_type, ir.PointerType):
                    return self.string_type
                else:
                    return self.int_type
            elif node.operator in ["EQUAL", "LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL", "AND", "OR"]:
                return self.bool_type
        elif isinstance(node, UnaryNode):
            rhs_type = self._infer_type(node.rhs, symbol_table, recursion_stack)
            if node.operator == "NOT":
                return self.bool_type
            elif node.operator == "MINUS":
                return rhs_type
        elif isinstance(node, CallNode):
            callee_name = node.callee.value if isinstance(node.callee, AtomicNode) else node.callee
            if isinstance(node.callee, (str, AtomicNode)):
                if callee_name in symbol_table and isinstance(symbol_table[callee_name], ir.Function):
                    return symbol_table[callee_name].type.pointee.return_type
                else:
                    return self.int_type
            elif isinstance(node.callee, LambdaNode):
                return self._infer_type(node.callee.body, symbol_table, recursion_stack)
        elif isinstance(node, IfNode):
            then_type = self._infer_type(node.ifBody, symbol_table, recursion_stack)
            else_type = self._infer_type(node.elseBody, symbol_table, recursion_stack) if node.elseBody else then_type
            for cond, body in node.elseIfs:
                body_type = self._infer_type(body, symbol_table, recursion_stack)
                if isinstance(body_type, ir.DoubleType):
                    return self.float_type
            if then_type == else_type:
                return then_type
            elif isinstance(then_type, ir.DoubleType) or isinstance(else_type, ir.DoubleType):
                return self.float_type
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
        return self.int_type

    def _generate_if(self, node):
        condition = self._generate_expr(node.condition)
        condition_bool = self._convert_to_bool(condition)
        
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
            else_value = ir.Constant(self.int_type, 0)
            else_end_block = self.builder.block
            if not then_end_block.is_terminated:
                self.builder.branch(merge_block)
        
        self.builder.position_at_end(merge_block)
        
        common_type = then_value.type
        if then_value.type != else_value.type:
            if isinstance(then_value.type, ir.DoubleType) or isinstance(else_value.type, ir.DoubleType):
                common_type = self.float_type
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
                    else_value = ir.Constant(common_type, 0)
        
        phi = self.builder.phi(common_type, name="if_result")
        phi.add_incoming(then_value, then_end_block)
        phi.add_incoming(else_value, else_end_block)
        
        return phi

    def _generate_else_ifs(self, else_ifs, else_body, final_merge_block):
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
        result = None
        for expr in node.expressions:
            result = self._generate_expr(expr)
        if result is None:
            result = ir.Constant(self.int_type, 0)
        return result

    def _convert_to_bool(self, value):
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
                elif isinstance(var_item, ir.GlobalVariable):
                    # Load the value from global variables (e.g., double for pi)
                    return self.builder.load(var_item, name=f"load_{node.value}")
                else:
                    # For local variables, load from the alloca
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
        if isinstance(value.type, ir.IntType) and value.type.width > 1:
            return self.builder.sitofp(value, self.float_type, name="promote")
        return value

    def _generate_binary(self, node):
        self._dprint(f"Generating binary op: {node.operator} ({node.left}, {node.right})")
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
        self._dprint(f"Left operand: {left} (type: {left.type}), Right operand: {right} (type: {right.type})")

        # Handle string concatenation for i8* types
        if node.operator == "PLUS" and isinstance(left.type, ir.PointerType) and isinstance(right.type, ir.PointerType):
            # Declare necessary C library functions
            strlen = self.module.globals.get('strlen')
            if not strlen:
                strlen_type = ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer()])
                strlen = ir.Function(self.module, strlen_type, name="strlen")
            
            malloc = self.module.globals.get('malloc')
            if not malloc:
                malloc_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(64)])
                malloc = ir.Function(self.module, malloc_type, name="malloc")
            
            strcpy = self.module.globals.get('strcpy')
            if not strcpy:
                strcpy_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(8).as_pointer(), ir.IntType(8).as_pointer()])
                strcpy = ir.Function(self.module, strcpy_type, name="strcpy")
            
            strcat = self.module.globals.get('strcat')
            if not strcat:
                strcat_type = ir.FunctionType(ir.IntType(8).as_pointer(), [ir.IntType(8).as_pointer(), ir.IntType(8).as_pointer()])
                strcat = ir.Function(self.module, strcat_type, name="strcat")
            
            # Compute lengths of both strings
            left_len = self.builder.call(strlen, [left], name="left_len")
            right_len = self.builder.call(strlen, [right], name="right_len")
            
            # Total length = left_len + right_len + 1 (for null terminator)
            total_len = self.builder.add(left_len, right_len, name="total_len")
            total_len_plus_one = self.builder.add(total_len, ir.Constant(ir.IntType(64), 1), name="total_len_plus_one")
            
            # Allocate buffer
            new_str = self.builder.call(malloc, [total_len_plus_one], name="new_str")
            
            # Copy first string
            self.builder.call(strcpy, [new_str, left], name="strcpy_tmp")
            
            # Append second string
            result = self.builder.call(strcat, [new_str, right], name="strcat_tmp")
            
            return result

        # Determine result type for numeric operations
        if node.operator in ["PLUS", "MINUS", "MULTIPLY", "DIVIDE", "POWER"]:
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType) or node.operator == "DIVIDE":
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                result_type = self.float_type
            else:
                result_type = self.int_type
        elif node.operator in ["EQUAL", "LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL", "AND", "OR"]:
            result_type = self.bool_type
        else:
            raise Exception(f"Unsupported binary operator: {node.operator}")

        # Arithmetic operations
        if node.operator == "PLUS":
            if result_type == self.float_type:
                return self.builder.fadd(left, right, name="addtmp")
            else:
                return self.builder.add(left, right, name="addtmp")
        elif node.operator == "MINUS":
            if result_type == self.float_type:
                return self.builder.fsub(left, right, name="subtmp")
            else:
                return self.builder.sub(left, right, name="subtmp")
        elif node.operator == "MULTIPLY":
            if result_type == self.float_type:
                return self.builder.fmul(left, right, name="multmp")
            else:
                return self.builder.mul(left, right, name="multmp")
        elif node.operator == "DIVIDE":
            left = self._promote_to_float(left)
            right = self._promote_to_float(right)
            return self.builder.fdiv(left, right, name="divtmp")
        elif node.operator == "POWER":
            left = self._promote_to_float(left)
            right = self._promote_to_float(right)
            pow_func = self.module.globals.get('pow')
            if not pow_func:
                pow_type = ir.FunctionType(self.float_type, [self.float_type, self.float_type])
                pow_func = ir.Function(self.module, pow_type, name="pow")
            return self.builder.call(pow_func, [left, right], name="powtmp")

        # Comparison operations
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

        # Logical operations.
        elif node.operator == "AND":
            left_bool = self._convert_to_bool(left)
            right_bool = self._convert_to_bool(right)
            return self.builder.and_(left_bool, right_bool, name="andtmp")
        elif node.operator == "OR":
            left_bool = self._convert_to_bool(left)
            right_bool = self._convert_to_bool(right)
            return self.builder.or_(left_bool, right_bool, name="ortmp")

    def _create_global_string(self, text: str):
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
        self._dprint(f"Generating call: {node}")
        if isinstance(node.callee, AtomicNode) and node.callee.type == "identifier":
            callee_name = node.callee.value
            if callee_name == "print":
                return self._handle_print_call(node)
            if callee_name in self.symbol_table:
                var_item = self.symbol_table[callee_name]
                if isinstance(var_item, ir.Function):
                    func = var_item
                else:
                    func = self.builder.load(var_item, name=f"load_{callee_name}")
            else:
                raise Exception(f"Unknown function: {callee_name}")
        elif isinstance(node.callee, str):
            callee_name = node.callee
            if callee_name == "print":
                return self._handle_print_call(node)
            elif callee_name in self.symbol_table and isinstance(self.symbol_table[callee_name], ir.Function):
                func = self.symbol_table[callee_name]
            else:
                raise Exception(f"Unknown function: {callee_name}")
        elif isinstance(node.callee, LambdaNode):
            func = self._generate_lambda(node.callee)
        else:
            raise Exception(f"Unsupported callee type: {type(node.callee)}")

        args = []
        for i, arg_node in enumerate(node.args):
            arg_val = self._generate_expr(arg_node)
            param_type = func.type.pointee.args[i] if i < len(func.type.pointee.args) else self.int_type
            if isinstance(param_type, ir.DoubleType) and isinstance(arg_val.type, ir.IntType) and arg_val.type.width > 1:
                arg_val = self.builder.sitofp(arg_val, self.float_type, name="promote_arg")
            args.append(arg_val)
        return self.builder.call(func, args, name="calltmp")

    def _handle_print_call(self, node):
        self._dprint(f"Generating print call with args: {node.args}")
        printf = self.module.globals.get('printf')
        if not printf:
            voidptr_ty = ir.IntType(8).as_pointer()
            printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
            printf = ir.Function(self.module, printf_ty, name="printf")

        llvm_args = []
        fmt_parts = []

        for i, arg_node in enumerate(node.args):
            arg = self._generate_expr(arg_node)
            if isinstance(arg_node, AtomicNode) and arg_node.type == "string" and arg_node.value.strip() == "":
                continue  # Skip spacer strings
            args_list = arg if isinstance(arg, list) else [arg]
            for j, a in enumerate(args_list):
                self._dprint(f"Print arg: {a} (type: {a.type})")
                if isinstance(a.type, ir.DoubleType):
                    fmt_parts.append("%.15f")
                    llvm_args.append(a)
                elif isinstance(a.type, ir.IntType) and a.type.width == 1:
                    fmt_parts.append("%s")
                    true_str = self._create_global_string("true")
                    false_str = self._create_global_string("false")
                    result_ptr = self.builder.select(a, true_str, false_str, name="bool_str")
                    llvm_args.append(result_ptr)
                elif isinstance(a.type, ir.IntType):
                    fmt_parts.append("%d")
                    llvm_args.append(a)
                elif isinstance(a.type, ir.PointerType):
                    fmt_parts.append("%s")
                    llvm_args.append(a)
                else:
                    raise Exception(f"Unsupported print argument type: {a.type}")
                    
                if j < len(args_list) - 1 or i < len(node.args) - 1:
                    fmt_parts.append(" ")

        fmt = "".join(fmt_parts) + "\n"
        fmt_ptr = self._create_global_string(fmt)
        self._dprint(f"Print format string: {fmt}")
        
        return self.builder.call(printf, [fmt_ptr] + llvm_args, name="calltmp")