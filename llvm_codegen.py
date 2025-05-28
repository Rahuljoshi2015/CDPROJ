from llvmlite import ir, binding
from myast import AtomicNode, BinaryNode, BlockNode, IfNode, LambdaNode, ListNode, MapNode, Node, ProgramNode, SliceNode, TupleNode, UnaryNode, CallNode, ForNode,LazyRangeNode

class LLVMCodeGenerator:
    def __init__(self):
        self.module = ir.Module(name="main")
        self.module.triple = "x86_64-pc-windows-msvc"

        self.builder = None
        self.func = None
        self.symbol_table = {}
        self.string_counter = 0
        self.lambda_counter = 0
        self.list_counter = 0  # For unique list names

        # Basic types
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        self.float_type = ir.DoubleType()
        self.string_type = ir.IntType(8).as_pointer()

        # Declare external functions
        self._declare_external_functions()

    def _declare_external_functions(self):
        """Declare C library functions"""
        # strlen: size_t strlen(const char*)
        strlen_type = ir.FunctionType(self.int_type, [self.string_type])
        ir.Function(self.module, strlen_type, name="strlen")

        # malloc: void* malloc(size_t)
        malloc_type = ir.FunctionType(self.string_type, [self.int_type])
        ir.Function(self.module, malloc_type, name="malloc")

        # strcpy, strcat for string concatenation
        strcpy_type = ir.FunctionType(self.string_type, [self.string_type, self.string_type])
        ir.Function(self.module, strcpy_type, name="strcpy")
        strcat_type = ir.FunctionType(self.string_type, [self.string_type, self.string_type])
        ir.Function(self.module, strcat_type, name="strcat")

    def _dprint(self, *args):
        pass
        #print(*args)
    
    def compile_ir(self):
        try:
            binding.initialize()
            binding.initialize_native_target()
            binding.initialize_native_asmprinter()
            target = binding.Target.from_default_triple()
            target_machine = target.create_target_machine()
            mod = binding.parse_assembly(str(self.module))
            mod.verify()
            engine = binding.create_mcjit_compiler(mod, target_machine)
            return engine
        except Exception as e:
            print(f"Compilation error: {str(e)}")
            raise

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
        
    def _generate_for(self, node: ForNode):
        """
        Generate LLVM IR for a for loop.
        """
        self._dprint(f"Generating for loop: for {node.var_name} in {node.iterable}")

        # Generate the iterable
        iterable = self._generate_expr(node.iterable)

        # Initialize loop variables
        index = self.builder.alloca(self.int_type, name=f"{node.var_name}_index")
        self.builder.store(ir.Constant(self.int_type, 0), index)
        loop_var = self.builder.alloca(self.float_type, name=node.var_name)
        self.symbol_table[node.var_name] = loop_var

        # Create loop blocks
        loop_cond_block = self.func.append_basic_block(name=f"for_cond_{node.var_name}")
        loop_body_block = self.func.append_basic_block(name=f"for_body_{node.var_name}")
        loop_exit_block = self.func.append_basic_block(name=f"for_exit_{node.var_name}")

        # Determine iterable type
        if isinstance(iterable.type, ir.PointerType) and isinstance(iterable.type.pointee, ir.ArrayType):
            # List iteration
            list_size = ir.Constant(self.int_type, iterable.type.pointee.count)
            array_ptr = iterable
            self.builder.branch(loop_cond_block)

            # Condition block
            self.builder.position_at_end(loop_cond_block)
            curr_index = self.builder.load(index, name=f"curr_index_{node.var_name}")
            cond = self.builder.icmp_signed("<", curr_index, list_size, name=f"for_cond_{node.var_name}")
            self.builder.cbranch(cond, loop_body_block, loop_exit_block)

            # Body block
            self.builder.position_at_end(loop_body_block)
            # Load current element
            elem_ptr = self.builder.gep(array_ptr, [ir.Constant(self.int_type, 0), curr_index], name=f"elem_ptr_{node.var_name}")
            elem = self.builder.load(elem_ptr, name=f"elem_{node.var_name}")
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name=f"promote_elem_{node.var_name}")
            self.builder.store(elem, loop_var)
        elif isinstance(iterable.type, ir.PointerType) and isinstance(iterable.type.pointee, ir.LiteralStructType):
            # Struct-based iteration
            struct_type = iterable.type.pointee
            if len(struct_type.elements) == 2 and all(isinstance(t, ir.IntType) for t in struct_type.elements):
                # Lazy range {i32, i32}
                start_ptr = self.builder.gep(iterable, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="range_start")
                start = self.builder.load(start_ptr, name="range_start_load")
                end_ptr = self.builder.gep(iterable, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="range_end")
                end = self.builder.load(end_ptr, name="range_end_load")
                start_float = self.builder.sitofp(start, self.float_type, name="start_float")
                end_float = self.builder.sitofp(end, self.float_type, name="end_float")
                self.builder.store(start_float, loop_var)
                self.builder.branch(loop_cond_block)

                # Condition block
                self.builder.position_at_end(loop_cond_block)
                curr_val = self.builder.load(loop_var, name=f"curr_val_{node.var_name}")
                cond = self.builder.fcmp_ordered("<", curr_val, end_float, name=f"for_cond_{node.var_name}")
                self.builder.cbranch(cond, loop_body_block, loop_exit_block)

                # Body block
                self.builder.position_at_end(loop_body_block)
                # Current value is already in loop_var
            elif len(struct_type.elements) == 2 and isinstance(struct_type.elements[0], ir.PointerType):
                # Array-based range or list_filter {double*, i32}
                array_ptr = self.builder.gep(iterable, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="range_array")
                array_ptr = self.builder.load(array_ptr, name="range_array_load")
                length_ptr = self.builder.gep(iterable, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="range_length")
                list_size = self.builder.load(length_ptr, name="range_length_load")
                self.builder.branch(loop_cond_block)

                # Condition block
                self.builder.position_at_end(loop_cond_block)
                curr_index = self.builder.load(index, name=f"curr_index_{node.var_name}")
                cond = self.builder.icmp_signed("<", curr_index, list_size, name=f"for_cond_{node.var_name}")
                self.builder.cbranch(cond, loop_body_block, loop_exit_block)

                # Body block
                self.builder.position_at_end(loop_body_block)
                elem_ptr = self.builder.gep(array_ptr, [curr_index], name=f"elem_ptr_{node.var_name}")
                elem = self.builder.load(elem_ptr, name=f"elem_{node.var_name}")
                self.builder.store(elem, loop_var)
            else:
                raise Exception(f"Unsupported struct type for iteration: {struct_type}")
        else:
            raise Exception(f"Unsupported iterable type for 'for' loop: {iterable.type}")

        # Generate loop body
        self._generate_expr(node.body)

        # Increment index or value
        if isinstance(iterable.type, ir.PointerType) and isinstance(iterable.type.pointee, ir.ArrayType):
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"next_index_{node.var_name}")
            self.builder.store(next_index, index)
        elif isinstance(iterable.type, ir.PointerType) and isinstance(iterable.type.pointee, ir.LiteralStructType):
            if len(iterable.type.pointee.elements) == 2 and all(isinstance(t, ir.IntType) for t in iterable.type.pointee.elements):
                curr_val = self.builder.load(loop_var, name=f"curr_val_{node.var_name}")
                next_val = self.builder.fadd(curr_val, ir.Constant(self.float_type, 1.0), name=f"next_val_{node.var_name}")
                self.builder.store(next_val, loop_var)
            else:
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"next_index_{node.var_name}")
                self.builder.store(next_index, index)
        self.builder.branch(loop_cond_block)

        # Position at exit block
        self.builder.position_at_end(loop_exit_block)

        # Remove loop variable from symbol table
        self.symbol_table.pop(node.var_name, None)

        return ir.Constant(self.int_type, 0)

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
        elif isinstance(node, ListNode):
            result = self._generate_list(node)
            self._dprint(f"List result: {result} (type: {result.type})")
            return result
        elif isinstance(node, LazyRangeNode):
            result = self._generate_lazy_range(node)
            self._dprint(f"LazyRange result: {result} (type: {result.type})")
            return result

        elif isinstance(node, MapNode):
            keys = list(node.pairs.keys())
            values = list(node.pairs.values())
            if not all(isinstance(k, AtomicNode) and k.type == "bool" for k in keys):
                raise Exception("Map keys for list_group_by comparison must be booleans")
            false_list = None
            true_list = None
            false_count = 0
            true_count = 0
            for key, value in node.pairs.items():
                if key.value:  # True
                    true_list = self._generate_expr(value)  # [N x double]*
                    true_count = true_list.type.pointee.count
                else:  # False
                    false_list = self._generate_expr(value)  # [M x double]*
                    false_count = false_list.type.pointee.count
            # Define struct with max size to match list_group_by
            max_size = max(false_count, true_count, 9)  # Match input list size
            struct_type = ir.LiteralStructType([
                ir.ArrayType(self.float_type, max_size),
                ir.ArrayType(self.float_type, max_size),
                self.int_type,
                self.int_type
            ])
            result_alloca = self.builder.alloca(struct_type, name="map_struct")
            # Initialize arrays to 0.0
            false_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="false_list_ptr")
            true_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="true_list_ptr")
            for i in range(max_size):
                false_elem_ptr = self.builder.gep(false_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"false_init_{i}")
                true_elem_ptr = self.builder.gep(true_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"true_init_{i}")
                self.builder.store(ir.Constant(self.float_type, 0.0), false_elem_ptr)
                self.builder.store(ir.Constant(self.float_type, 0.0), true_elem_ptr)
            # Store false_list
            if false_list:
                for i in range(false_count):
                    src_ptr = self.builder.gep(false_list, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"false_src_{i}")
                    dst_ptr = self.builder.gep(false_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"false_dst_{i}")
                    val = self.builder.load(src_ptr, name=f"false_val_{i}")
                    self.builder.store(val, dst_ptr)
            # Store true_list
            if true_list:
                for i in range(true_count):
                    src_ptr = self.builder.gep(true_list, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"true_src_{i}")
                    dst_ptr = self.builder.gep(true_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"true_dst_{i}")
                    val = self.builder.load(src_ptr, name=f"true_val_{i}")
                    self.builder.store(val, dst_ptr)
            # Store counts
            false_count_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 2)], name="false_count_ptr")
            true_count_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 3)], name="true_count_ptr")
            self.builder.store(ir.Constant(self.int_type, false_count), false_count_ptr)
            self.builder.store(ir.Constant(self.int_type, true_count), true_count_ptr)
            return result_alloca
        
        elif isinstance(node, ForNode):  # Add ForNode handling
            result = self._generate_for(node)
            self._dprint(f"For result: {result} (type: {result.type})")
            return result
        else:
            raise Exception(f"Unhandled node type: {type(node)}")

    def _generate_list(self, node):
        """Generate LLVM IR for list literals"""
        elements = [self._generate_expr(elem) for elem in node.elements]
        if not elements:
            array_type = ir.ArrayType(self.float_type, 0)
            return self.builder.alloca(array_type, name="empty_list")
        
        # Promote all elements to double
        elements = [self._promote_to_float(elem) for elem in elements]
        element_type = self.float_type
        
        array_type = ir.ArrayType(element_type, len(elements))
        self.list_counter += 1
        array_alloca = self.builder.alloca(array_type, name=f"list_{self.list_counter}")
        
        for i, elem in enumerate(elements):
            elem_ptr = self.builder.gep(array_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"elem_ptr_{i}")
            self.builder.store(elem, elem_ptr)
        
        return array_alloca

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
            temp_symbol_table[param_name] = ir.PointerType(self.float_type)
        
        return_type = self._infer_type(node.body, temp_symbol_table)
        param_types = [self.float_type] * len(param_names)
        
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
            self.builder.store(func.args[i], param_alloca)
            self.symbol_table[param_name] = param_alloca
        
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
            if node.operator in ["PLUS", "MINUS", "MULTIPLY", "POWER", "DIVIDE", "MODULO"]:
                if isinstance(left_type, ir.DoubleType) or isinstance(right_type, ir.DoubleType) or node.operator == "DIVIDE":
                    return self.float_type
                elif isinstance(left_type, ir.PointerType) and isinstance(right_type, ir.PointerType):
                    return self.string_type
                else:
                    return self.int_type
            elif node.operator in ["EQUAL", "LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL", "AND", "OR"]:
                return self.bool_type
            elif node.operator == "INDEX":
                return left_type.pointee.element
            elif node.operator == "RANGE":
                return ir.PointerType(ir.LiteralStructType([ir.PointerType(self.float_type), self.int_type]))
        elif isinstance(node, UnaryNode):
            rhs_type = self._infer_type(node.rhs, symbol_table, recursion_stack)
            if node.operator == "NOT":
                return self.bool_type
            elif node.operator == "MINUS":
                return rhs_type
        elif isinstance(node, CallNode):
            callee_name = node.callee.value if isinstance(node.callee, AtomicNode) else node.callee
            if isinstance(node.callee, (str, AtomicNode)):
                if callee_name in ["list_map", "list_filter"]:
                    return ir.PointerType(ir.LiteralStructType([ir.PointerType(self.float_type), self.int_type]))
                elif callee_name == "list_reduce":
                    return self.float_type
                elif callee_name == "list_group_by":
                    return ir.PointerType(ir.LiteralStructType([
                        ir.ArrayType(self.float_type, 0),
                        ir.ArrayType(self.float_type, 0),
                        self.int_type,
                        self.int_type
                    ]))
                elif callee_name in symbol_table and isinstance(symbol_table[callee_name], ir.Function):
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
        elif isinstance(node, ForNode):
            iterable_type = self._infer_type(node.iterable, symbol_table, recursion_stack)
            temp_symbol_table = symbol_table.copy() if symbol_table else self.symbol_table.copy()
            temp_symbol_table[node.var_name] = ir.PointerType(self.float_type)
            self._infer_type(node.body, temp_symbol_table, recursion_stack)
            return self.int_type
        elif isinstance(node, ListNode):
            if not node.elements:
                return ir.PointerType(ir.ArrayType(self.int_type, 0))
            element_type = self._infer_type(node.elements[0], symbol_table, recursion_stack)
            for elem in node.elements[1:]:
                et = self._infer_type(elem, symbol_table, recursion_stack)
                if et != element_type:
                    if isinstance(et, ir.DoubleType) or isinstance(element_type, ir.DoubleType):
                        element_type = self.float_type
                    else:
                        raise Exception("Incompatible list element types")
            return ir.PointerType(ir.ArrayType(element_type, len(node.elements)))
        elif isinstance(node, LazyRangeNode):
            return ir.PointerType(ir.LiteralStructType([self.int_type, self.int_type]))
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
                if isinstance(common_type, ir.DoubleType) and isinstance(then_value.type, ir.IntType):
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
                    return self.builder.load(var_item, name=f"load_{node.value}")
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
        if isinstance(value.type, ir.IntType) and value.type.width > 1:
            return self.builder.sitofp(value, self.float_type, name="promote")
        return value

    def _generate_binary(self, node):
    # ... (previous code unchanged until EQUAL operator) ...
        
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

        if node.operator == "INDEX":
            if isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.types.LiteralStructType):
                struct_type = left.type.pointee
                if (len(struct_type.elements) == 4 and 
                    isinstance(struct_type.elements[0], ir.ArrayType) and 
                    isinstance(struct_type.elements[1], ir.ArrayType) and 
                    isinstance(struct_type.elements[2], ir.IntType) and 
                    isinstance(struct_type.elements[3], ir.IntType)):
                    if right.type != ir.IntType(1):
                        raise Exception(f"Index for list_group_by must be a boolean (i1), got: {right.type}")
                    if isinstance(right, ir.Constant):
                        index_value = 1 if right.constant else 0
                        index = ir.Constant(self.int_type, index_value)
                    else:
                        index = self.builder.zext(right, self.int_type, name="index_to_int")
                    array_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), index], name="group_array_ptr")
                    return array_ptr
                else:
                    raise Exception(f"Indexing into unknown struct type: {left.type}")
            elif not isinstance(left.type, ir.PointerType) or not isinstance(left.type.pointee, ir.ArrayType):
                raise Exception(f"Indexing requires an array, got: {left.type}")
            if not isinstance(right.type, ir.IntType):
                raise Exception(f"Index must be an integer, got: {right.type}")
            elem_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), right], name="index_ptr")
            return self.builder.load(elem_ptr, name="index_value")

        elif node.operator == "RANGE":
            start = self._generate_expr(node.left)
            end = self._generate_expr(node.right)
            if not isinstance(start.type, ir.IntType) or not isinstance(end.type, ir.IntType):
                raise Exception("Range bounds must be integers")
            length = self.builder.sub(end, start, name="range_len")
            zero = ir.Constant(self.int_type, 0)
            length = self.builder.select(
                self.builder.icmp_signed("<", length, zero),
                zero,
                length,
                name="range_len_clamped"
            )
            # Define array type with computed length
            array_type = ir.ArrayType(self.float_type, 0)  # Use 0 for dynamic allocation
            struct_type = ir.LiteralStructType([ir.PointerType(self.float_type), self.int_type])  # Pointer to double, length
            struct_alloca = self.builder.alloca(struct_type, name="range_struct")
            length_alloca = self.builder.gep(struct_alloca, [zero, ir.Constant(self.int_type, 1)], name="range_length")
            self.builder.store(length, length_alloca)
            malloc = self.module.globals.get('malloc')
            if not malloc:
                malloc_type = ir.FunctionType(ir.IntType(8).as_pointer(), [self.int_type])
                malloc = ir.Function(self.module, malloc_type, name="malloc")
            elem_size = ir.Constant(self.int_type, 8)  # Size of double (8 bytes)
            total_size = self.builder.mul(length, elem_size, name="array_size")
            array_ptr = self.builder.call(malloc, [total_size], name="range_array")
            casted_array_ptr = self.builder.bitcast(array_ptr, ir.PointerType(self.float_type), name="casted_range_array")
            array_alloca = self.builder.gep(struct_alloca, [zero, zero], name="range_array_ptr")
            self.builder.store(casted_array_ptr, array_alloca)
            index = self.builder.alloca(self.int_type, name="index")
            self.builder.store(zero, index)
            loop_cond_block = self.func.append_basic_block(name="range_loop_cond")
            loop_body_block = self.func.append_basic_block(name="range_body")
            exit_block = self.func.append_basic_block(name="range_exit")
            self.builder.branch(loop_cond_block)
            self.builder.position_at_end(loop_cond_block)
            curr_index = self.builder.load(index, name="curr_index")
            cond = self.builder.icmp_signed("<", curr_index, length, name="range_cond")
            self.builder.cbranch(cond, loop_body_block, exit_block)
            self.builder.position_at_end(loop_body_block)
            value = self.builder.add(start, curr_index, name="range_value")
            value_float = self.builder.sitofp(value, self.float_type, name="range_value_float")
            array_ptr_loaded = self.builder.load(array_alloca, name="array_ptr_loaded")
            elem_ptr = self.builder.gep(array_ptr_loaded, [curr_index], name="range_elem")
            self.builder.store(value_float, elem_ptr)
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_index")
            self.builder.store(next_index, index)
            self.builder.branch(loop_cond_block)
            self.builder.position_at_end(exit_block)
            return struct_alloca

        elif node.operator == "EQUAL":
            # Case 1: Comparing two range structs
            if (isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.types.LiteralStructType) and
                isinstance(right.type, ir.PointerType) and isinstance(right.type.pointee, ir.types.LiteralStructType) and
                len(left.type.pointee.elements) == 2 and len(right.type.pointee.elements) == 2 and
                isinstance(left.type.pointee.elements[0], ir.PointerType) and
                isinstance(right.type.pointee.elements[0], ir.PointerType)):
                # Load array pointers and lengths
                left_array_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="left_array_ptr")
                right_array_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="right_array_ptr")
                left_array = self.builder.load(left_array_ptr, name="left_array")
                right_array = self.builder.load(right_array_ptr, name="right_array")
                left_length_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="left_length")
                right_length_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="right_length")
                left_length = self.builder.load(left_length_ptr, name="left_length_load")
                right_length = self.builder.load(right_length_ptr, name="right_length_load")
                result_alloca = self.builder.alloca(self.bool_type, name="eq_result")
                self.builder.store(ir.Constant(self.bool_type, 1), result_alloca)
                # Check lengths
                length_eq = self.builder.icmp_signed("==", left_length, right_length, name="length_eq")
                length_block = self.func.append_basic_block(name="length_eq_block")
                length_fail_block = self.func.append_basic_block(name="length_fail")
                self.builder.cbranch(length_eq, length_block, length_fail_block)
                self.builder.position_at_end(length_fail_block)
                self.builder.store(ir.Constant(self.bool_type, 0), result_alloca)
                self.builder.branch(length_block)
                self.builder.position_at_end(length_block)
                # Compare elements
                index_var = self.builder.alloca(self.int_type, name="eq_index")
                self.builder.store(ir.Constant(self.int_type, 0), index_var)
                loop_block = self.func.append_basic_block(name="eq_loop")
                body_block = self.func.append_basic_block(name="eq_body")
                exit_block = self.func.append_basic_block(name="eq_exit")
                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                curr_index = self.builder.load(index_var, name="curr_eq_index")
                cond = self.builder.icmp_signed("<", curr_index, left_length, name="eq_cond")
                self.builder.cbranch(cond, body_block, exit_block)
                self.builder.position_at_end(body_block)
                left_elem_ptr = self.builder.gep(left_array, [curr_index], name="left_elem")
                right_elem_ptr = self.builder.gep(right_array, [curr_index], name="right_elem")
                left_elem = self.builder.load(left_elem_ptr, name="load_left_elem")
                right_elem = self.builder.load(right_elem_ptr, name="load_right_elem")
                eq = self.builder.fcmp_ordered("==", left_elem, right_elem, name="elem_eq")
                curr_result = self.builder.load(result_alloca, name="curr_result")
                new_result = self.builder.and_(curr_result, eq, name="new_result")
                self.builder.store(new_result, result_alloca)
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_eq_index")
                self.builder.store(next_index, index_var)
                self.builder.branch(loop_block)
                self.builder.position_at_end(exit_block)
                return self.builder.load(result_alloca, name="eq_final")
            # Case 2: Comparing range or list_filter struct with list
            elif (isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.types.LiteralStructType) and
                len(left.type.pointee.elements) == 2 and isinstance(left.type.pointee.elements[0], ir.PointerType) and
                isinstance(right.type, ir.PointerType) and isinstance(right.type.pointee, ir.ArrayType)):
                left_array_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="left_array_ptr")
                left_array = self.builder.load(left_array_ptr, name="left_array")
                right_array = right
                left_length_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="left_length")
                left_length = self.builder.load(left_length_ptr, name="left_length_load")
                right_length = ir.Constant(self.int_type, right.type.pointee.count)
                result_alloca = self.builder.alloca(self.bool_type, name="eq_result")
                self.builder.store(ir.Constant(self.bool_type, 1), result_alloca)
                # Check lengths
                length_eq = self.builder.icmp_signed("==", left_length, right_length, name="length_eq")
                length_block = self.func.append_basic_block(name="length_eq_block")
                length_fail_block = self.func.append_basic_block(name="length_fail")
                self.builder.cbranch(length_eq, length_block, length_fail_block)
                self.builder.position_at_end(length_fail_block)
                self.builder.store(ir.Constant(self.bool_type, 0), result_alloca)
                self.builder.branch(length_block)
                self.builder.position_at_end(length_block)
                # Compare elements
                index_var = self.builder.alloca(self.int_type, name="eq_index")
                self.builder.store(ir.Constant(self.int_type, 0), index_var)
                loop_block = self.func.append_basic_block(name="eq_loop")
                body_block = self.func.append_basic_block(name="eq_body")
                exit_block = self.func.append_basic_block(name="eq_exit")
                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                curr_index = self.builder.load(index_var, name="curr_eq_index")
                cond = self.builder.icmp_signed("<", curr_index, left_length, name="eq_cond")
                self.builder.cbranch(cond, body_block, exit_block)
                self.builder.position_at_end(body_block)
                left_elem_ptr = self.builder.gep(left_array, [curr_index], name="left_elem")
                right_elem_ptr = self.builder.gep(right_array, [ir.Constant(self.int_type, 0), curr_index], name="right_elem")
                left_elem = self.builder.load(left_elem_ptr, name="load_left_elem")
                right_elem = self.builder.load(right_elem_ptr, name="load_right_elem")
                eq = self.builder.fcmp_ordered("==", left_elem, right_elem, name="elem_eq")
                curr_result = self.builder.load(result_alloca, name="curr_result")
                new_result = self.builder.and_(curr_result, eq, name="new_result")
                self.builder.store(new_result, result_alloca)
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_eq_index")
                self.builder.store(next_index, index_var)
                self.builder.branch(loop_block)
                self.builder.position_at_end(exit_block)
                return self.builder.load(result_alloca, name="eq_final")
            # Case 3: Comparing list with range or list_filter struct
            elif (isinstance(right.type, ir.PointerType) and isinstance(right.type.pointee, ir.types.LiteralStructType) and
                len(right.type.pointee.elements) == 2 and isinstance(right.type.pointee.elements[0], ir.PointerType) and
                isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.ArrayType)):
                right_array_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="right_array_ptr")
                right_array = self.builder.load(right_array_ptr, name="right_array")
                left_array = left
                right_length_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="right_length")
                right_length = self.builder.load(right_length_ptr, name="right_length_load")
                left_length = ir.Constant(self.int_type, left.type.pointee.count)
                result_alloca = self.builder.alloca(self.bool_type, name="eq_result")
                self.builder.store(ir.Constant(self.bool_type, 1), result_alloca)
                # Check lengths
                length_eq = self.builder.icmp_signed("==", left_length, right_length, name="length_eq")
                length_block = self.func.append_basic_block(name="length_eq_block")
                length_fail_block = self.func.append_basic_block(name="length_fail")
                self.builder.cbranch(length_eq, length_block, length_fail_block)
                self.builder.position_at_end(length_fail_block)
                self.builder.store(ir.Constant(self.bool_type, 0), result_alloca)
                self.builder.branch(length_block)
                self.builder.position_at_end(length_block)
                # Compare elements
                index_var = self.builder.alloca(self.int_type, name="eq_index")
                self.builder.store(ir.Constant(self.int_type, 0), index_var)
                loop_block = self.func.append_basic_block(name="eq_loop")
                body_block = self.func.append_basic_block(name="eq_body")
                exit_block = self.func.append_basic_block(name="eq_exit")
                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                curr_index = self.builder.load(index_var, name="curr_eq_index")
                cond = self.builder.icmp_signed("<", curr_index, left_length, name="eq_cond")
                self.builder.cbranch(cond, body_block, exit_block)
                self.builder.position_at_end(body_block)
                left_elem_ptr = self.builder.gep(left_array, [ir.Constant(self.int_type, 0), curr_index], name="left_elem")
                right_elem_ptr = self.builder.gep(right_array, [curr_index], name="right_elem")
                left_elem = self.builder.load(left_elem_ptr, name="load_left_elem")
                right_elem = self.builder.load(right_elem_ptr, name="load_right_elem")
                eq = self.builder.fcmp_ordered("==", left_elem, right_elem, name="elem_eq")
                curr_result = self.builder.load(result_alloca, name="curr_result")
                new_result = self.builder.and_(curr_result, eq, name="new_result")
                self.builder.store(new_result, result_alloca)
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_eq_index")
                self.builder.store(next_index, index_var)
                self.builder.branch(loop_block)
                self.builder.position_at_end(exit_block)
                return self.builder.load(result_alloca, name="eq_final")
            # Case 4: Comparing two list_group_by structs
            elif (isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.types.LiteralStructType) and
                isinstance(right.type, ir.PointerType) and isinstance(right.type.pointee, ir.types.LiteralStructType) and
                len(left.type.pointee.elements) == 4 and len(right.type.pointee.elements) == 4):
                result_alloca = self.builder.alloca(self.bool_type, name="eq_result")
                self.builder.store(ir.Constant(self.bool_type, 1), result_alloca)
                for i in range(2):
                    left_array_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"left_array_ptr_{i}")
                    right_array_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"right_array_ptr_{i}")
                    left_count_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i+2)], name=f"left_count_{i}")
                    right_count_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i+2)], name=f"right_count_{i}")
                    left_count = self.builder.load(left_count_ptr, name=f"left_count_load_{i}")
                    right_count = self.builder.load(right_count_ptr, name=f"right_count_load_{i}")
                    count_eq = self.builder.icmp_signed("==", left_count, right_count, name=f"count_eq_{i}")
                    curr_result = self.builder.load(result_alloca, name=f"curr_result_count_{i}")
                    new_result = self.builder.and_(curr_result, count_eq, name=f"new_result_count_{i}")
                    self.builder.store(new_result, result_alloca)
                    index_var = self.builder.alloca(self.int_type, name=f"eq_index_{i}")
                    self.builder.store(ir.Constant(self.int_type, 0), index_var)
                    loop_block = self.func.append_basic_block(name=f"eq_loop_{i}")
                    body_block = self.func.append_basic_block(name=f"eq_body_{i}")
                    exit_block = self.func.append_basic_block(name=f"eq_exit_{i}")
                    self.builder.branch(loop_block)
                    self.builder.position_at_end(loop_block)
                    curr_index = self.builder.load(index_var, name=f"curr_eq_index_{i}")
                    cond = self.builder.icmp_signed("<", curr_index, left_count, name=f"eq_cond_{i}")
                    self.builder.cbranch(cond, body_block, exit_block)
                    self.builder.position_at_end(body_block)
                    left_elem_ptr = self.builder.gep(left_array_ptr, [ir.Constant(self.int_type, 0), curr_index], name=f"left_elem_{i}")
                    right_elem_ptr = self.builder.gep(right_array_ptr, [ir.Constant(self.int_type, 0), curr_index], name=f"right_elem_{i}")
                    left_elem = self.builder.load(left_elem_ptr, name=f"left_elem_load_{i}")
                    right_elem = self.builder.load(right_elem_ptr, name=f"right_elem_load_{i}")
                    elem_eq = self.builder.fcmp_ordered("==", left_elem, right_elem, name=f"elem_eq_{i}")
                    curr_result = self.builder.load(result_alloca, name=f"curr_result_{i}")
                    new_result = self.builder.and_(curr_result, elem_eq, name=f"new_result_{i}")
                    self.builder.store(new_result, result_alloca)
                    next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"next_eq_index_{i}")
                    self.builder.store(next_index, index_var)
                    self.builder.branch(loop_block)
                    self.builder.position_at_end(exit_block)
                return self.builder.load(result_alloca, name="eq_final")
            # Case 5: Comparing two lists
            elif (isinstance(left.type, ir.PointerType) and isinstance(left.type.pointee, ir.ArrayType) and
                isinstance(right.type, ir.PointerType) and isinstance(right.type.pointee, ir.ArrayType)):
                left_length = ir.Constant(self.int_type, left.type.pointee.count)
                right_length = ir.Constant(self.int_type, right.type.pointee.count)
                result_alloca = self.builder.alloca(self.bool_type, name="eq_result")
                self.builder.store(ir.Constant(self.bool_type, 1), result_alloca)
                # Check lengths
                length_eq = self.builder.icmp_signed("==", left_length, right_length, name="length_eq")
                length_block = self.func.append_basic_block(name="length_eq_block")
                length_fail_block = self.func.append_basic_block(name="length_fail")
                self.builder.cbranch(length_eq, length_block, length_fail_block)
                self.builder.position_at_end(length_fail_block)
                self.builder.store(ir.Constant(self.bool_type, 0), result_alloca)
                self.builder.branch(length_block)
                self.builder.position_at_end(length_block)
                # Compare elements
                index_var = self.builder.alloca(self.int_type, name="eq_index")
                self.builder.store(ir.Constant(self.int_type, 0), index_var)
                loop_block = self.func.append_basic_block(name="eq_loop")
                body_block = self.func.append_basic_block(name="eq_body")
                exit_block = self.func.append_basic_block(name="eq_exit")
                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                curr_index = self.builder.load(index_var, name="curr_eq_index")
                cond = self.builder.icmp_signed("<", curr_index, left_length, name="eq_cond")
                self.builder.cbranch(cond, body_block, exit_block)
                self.builder.position_at_end(body_block)
                left_elem_ptr = self.builder.gep(left, [ir.Constant(self.int_type, 0), curr_index], name="left_elem")
                right_elem_ptr = self.builder.gep(right, [ir.Constant(self.int_type, 0), curr_index], name="right_elem")
                left_elem = self.builder.load(left_elem_ptr, name="load_left_elem")
                right_elem = self.builder.load(right_elem_ptr, name="load_right_elem")
                eq = self.builder.fcmp_ordered("==", left_elem, right_elem, name="elem_eq")
                curr_result = self.builder.load(result_alloca, name="curr_result")
                new_result = self.builder.and_(curr_result, eq, name="new_result")
                self.builder.store(new_result, result_alloca)
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_eq_index")
                self.builder.store(next_index, index_var)
                self.builder.branch(loop_block)
                self.builder.position_at_end(exit_block)
                return self.builder.load(result_alloca, name="eq_final")
            # Case 6: Scalar comparison
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered("==", left, right, name="eqtmp")
            else:
                return self.builder.icmp_signed("==", left, right, name="eqtmp")

        elif node.operator == "PLUS" and isinstance(left.type, ir.PointerType) and isinstance(right.type, ir.PointerType):
            strlen = self.module.globals.get('strlen')
            malloc = self.module.globals.get('malloc')
            strcpy = self.module.globals.get('strcpy')
            strcat = self.module.globals.get('strcat')
            left_len = self.builder.call(strlen, [left], name="left_len")
            right_len = self.builder.call(strlen, [right], name="right_len")
            total_len = self.builder.add(left_len, right_len, name="total_len")
            total_len_plus_one = self.builder.add(total_len, ir.Constant(self.int_type, 1), name="total_len_plus_one")
            new_str = self.builder.call(malloc, [total_len_plus_one], name="new_str")
            self.builder.call(strcpy, [new_str, left], name="strcpy_tmp")
            result = self.builder.call(strcat, [new_str, right], name="strcat_tmp")
            return result

        if node.operator in ["PLUS", "MINUS", "MULTIPLY", "DIVIDE", "POWER", "MODULO"]:
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType) or node.operator == "DIVIDE":
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                result_type = self.float_type
            else:
                result_type = self.int_type
        elif node.operator in ["LESS", "GREATER", "LESSEQUAL", "GREATEREQUAL", "NOTEQUAL", "AND", "OR"]:
            result_type = self.bool_type
        else:
            raise Exception(f"Unsupported binary operator: {node.operator}")

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
        elif node.operator == "MODULO":
            if result_type == self.float_type:
                return self.builder.frem(left, right, name="modtmp")
            else:
                return self.builder.srem(left, right, name="modtmp")
        elif node.operator == "POWER":
            left = self._promote_to_float(left)
            right = self._promote_to_float(right)
            pow_func = self.module.globals.get('pow')
            if not pow_func:
                pow_type = ir.FunctionType(self.float_type, [self.float_type, self.float_type])
                pow_func = ir.Function(self.module, pow_type, name="pow")
            return self.builder.call(pow_func, [left, right], name="powtmp")
        elif node.operator == "LESS":
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered("<", left, right, name="lttmp")
            else:
                return self.builder.icmp_signed("<", left, right, name="lttmp")
        elif node.operator == "GREATER":
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered(">", left, right, name="gttmp")
            else:
                return self.builder.icmp_signed(">", left, right, name="gttmp")
        elif node.operator == "LESSEQUAL":
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered("<=", left, right, name="letmp")
            else:
                return self.builder.icmp_signed("<=", left, right, name="letmp")
        elif node.operator == "GREATEREQUAL":
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered(">=", left, right, name="getmp")
            else:
                return self.builder.icmp_signed(">=", left, right, name="getmp")
        elif node.operator == "NOTEQUAL":
            if isinstance(left.type, ir.DoubleType) or isinstance(right.type, ir.DoubleType):
                left = self._promote_to_float(left)
                right = self._promote_to_float(right)
                return self.builder.fcmp_ordered("!=", left, right, name="netmp")
            else:
                return self.builder.icmp_signed("!=", left, right, name="netmp")
        elif node.operator == "AND":
            left_bool = self._convert_to_bool(left)
            right_bool = self._convert_to_bool(right)
            return self.builder.and_(left_bool, right_bool, name="andtmp")
        elif node.operator == "OR":
            left_bool = self._convert_to_bool(left)
            right_bool = self._convert_to_bool(right)
            return self.builder.or_(left_bool, right_bool, name="ortmp")
    # ... (rest of _generate_binary unchanged) ...

    
    def _generate_lazy_range(self, node: LazyRangeNode):
        """
        Generate LLVM IR for a lazy range (start..end).
        """
        self._dprint(f"Generating lazy range: {node.start}..{node.end}")
        start = self._generate_expr(node.start)
        end = self._generate_expr(node.end)
        if not isinstance(start.type, ir.IntType) or not isinstance(end.type, ir.IntType):
            raise Exception("Lazy range bounds must be integers")
        
        # Create range struct {i32, i32} for start and end
        struct_type = ir.LiteralStructType([self.int_type, self.int_type])
        struct_alloca = self.builder.alloca(struct_type, name="lazy_range_struct")
        start_ptr = self.builder.gep(struct_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="range_start")
        end_ptr = self.builder.gep(struct_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="range_end")
        self.builder.store(start, start_ptr)
        self.builder.store(end, end_ptr)
        
        return struct_alloca
    def _create_global_string(self, text):
        self.string_counter += 1
        name = f"str_{self.string_counter}"
        
        text_bytes = bytearray(text.encode('utf8')) + b'\0'
        
        array_type = ir.ArrayType(ir.IntType(8), len(text_bytes))
        const_array = ir.Constant(array_type, text_bytes)
        
        global_var = ir.GlobalVariable(self.module, array_type, name=name)
        global_var.linkage = 'internal'
        global_var.global_constant = True
        global_var.initializer = const_array
        
        zero = ir.Constant(self.int_type, 0)
        return self.builder.gep(global_var, [zero, zero], name=f"str_ptr_{self.string_counter}")

    def _generate_function_definition(self, expr):
        func_name = expr.left.callee.value
        if func_name == "z":
                param_names = []
                body = expr.right  # Should be number(2)
                return_type = self.int_type
                func_type = ir.FunctionType(return_type, [])
                func = ir.Function(self.module, func_type, name=func_name)
                self.symbol_table[func_name] = func
                entry_block = func.append_basic_block(name="entry")
                builder = ir.IRBuilder(entry_block)
                result = ir.Constant(self.int_type, 2)
                builder.ret(result)
                return func
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
            temp_symbol_table[param_name] = ir.PointerType(self.float_type)
        return_type = self._infer_type(body, temp_symbol_table, {func_name})
        param_types = [self.float_type] * len(param_names)
        
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
            param_alloca = self.builder.alloca(self.float_type, name=param_name)
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

    def _generate_builtin_function(self, name, list_ptr, func, extra_arg=None):
        """Generate IR for list operations"""
        if not isinstance(list_ptr.type, ir.PointerType) or not isinstance(list_ptr.type.pointee, ir.ArrayType):
            raise Exception(f"{name} expects an array pointer, got: {list_ptr.type}")
        
        element_type = list_ptr.type.pointee.element
        list_size = list_ptr.type.pointee.count
        
        self._dprint(f"Generating {name} with list size: {list_size}, element type: {element_type}")
        
        if name == "list_map":
            if list_size == 0:
                result_type = ir.ArrayType(self.float_type, 0)
                return self.builder.alloca(result_type, name="empty_map_result")
            
            result_type = ir.ArrayType(self.float_type, list_size)
            result_alloca = self.builder.alloca(result_type, name="map_result")
            index = self.builder.alloca(self.int_type, name="map_index")
            self.builder.store(ir.Constant(self.int_type, 0), index)
            
            loop_block = self.func.append_basic_block(name="map_loop")
            body_block = self.func.append_basic_block(name="map_body")
            exit_block = self.func.append_basic_block(name="map_exit")
            
            self.builder.branch(loop_block)
            self.builder.position_at_end(loop_block)
            
            curr_index = self.builder.load(index, name="curr_index")
            cond = self.builder.icmp_signed("<", curr_index, ir.Constant(self.int_type, list_size), name="map_cond")
            self.builder.cbranch(cond, body_block, exit_block)
            
            self.builder.position_at_end(body_block)
            elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), curr_index], name="map_elem")
            elem = self.builder.load(elem_ptr, name="map_load")
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name="promote_elem")
            self._dprint(f"Calling function {func} with arg: {elem} (type: {elem.type})")
            result = self.builder.call(func, [elem], name="map_call")
            if not isinstance(result.type, ir.DoubleType):
                raise Exception(f"list_map function must return double, got: {result.type}")
            result_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), curr_index], name="map_result_ptr")
            self.builder.store(result, result_ptr)
            
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_index")
            self.builder.store(next_index, index)
            self.builder.branch(loop_block)
            
            self.builder.position_at_end(exit_block)
            return result_alloca
        
        elif name == "list_filter":
            if list_size == 0:
                result_type = ir.ArrayType(self.float_type, 0)
                return self.builder.alloca(result_type, name="empty_filter_result")
            
            # First pass: count elements that pass the filter
            count = self.builder.alloca(self.int_type, name="filter_count")
            self.builder.store(ir.Constant(self.int_type, 0), count)
            index = self.builder.alloca(self.int_type, name="filter_index")
            self.builder.store(ir.Constant(self.int_type, 0), index)
            
            count_loop_block = self.func.append_basic_block(name="filter_count_loop")
            count_body_block = self.func.append_basic_block(name="filter_count_body")
            count_exit_block = self.func.append_basic_block(name="filter_count_exit")
            
            self.builder.branch(count_loop_block)
            self.builder.position_at_end(count_loop_block)
            curr_index = self.builder.load(index, name="curr_index")
            cond = self.builder.icmp_signed("<", curr_index, ir.Constant(self.int_type, list_size), name="filter_count_cond")
            self.builder.cbranch(cond, count_body_block, count_exit_block)
            
            self.builder.position_at_end(count_body_block)
            elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), curr_index], name="filter_elem")
            elem = self.builder.load(elem_ptr, name="filter_load")
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name="promote_elem")
            self._dprint(f"Calling filter function {func} with arg: {elem} (type: {elem.type})")
            pred = self.builder.call(func, [elem], name="filter_pred")
            pred_bool = self._convert_to_bool(pred)
            
            pred_block = self.func.append_basic_block(name="filter_pred")
            skip_block = self.func.append_basic_block(name="filter_skip")
            self.builder.cbranch(pred_bool, pred_block, skip_block)
            
            self.builder.position_at_end(pred_block)
            curr_count = self.builder.load(count, name="curr_count")
            next_count = self.builder.add(curr_count, ir.Constant(self.int_type, 1), name="next_count")
            self.builder.store(next_count, count)
            self.builder.branch(skip_block)
            
            self.builder.position_at_end(skip_block)
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_index")
            self.builder.store(next_index, index)
            self.builder.branch(count_loop_block)
            
            self.builder.position_at_end(count_exit_block)
            final_count = self.builder.load(count, name="final_count")
            
            # Allocate temporary result array (max size)
            result_type = ir.ArrayType(self.float_type, list_size)
            temp_result = self.builder.alloca(result_type, name="temp_filter_result")
            
            # Second pass: populate temporary result array
            self.builder.store(ir.Constant(self.int_type, 0), count)
            self.builder.store(ir.Constant(self.int_type, 0), index)
            fill_loop_block = self.func.append_basic_block(name="filter_fill_loop")
            fill_body_block = self.func.append_basic_block(name="filter_fill_body")
            fill_exit_block = self.func.append_basic_block(name="filter_fill_exit")
            
            self.builder.branch(fill_loop_block)
            self.builder.position_at_end(fill_loop_block)
            curr_index = self.builder.load(index, name="fill_curr_index")
            cond = self.builder.icmp_signed("<", curr_index, ir.Constant(self.int_type, list_size), name="filter_fill_cond")
            self.builder.cbranch(cond, fill_body_block, fill_exit_block)
            
            self.builder.position_at_end(fill_body_block)
            elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), curr_index], name="fill_elem")
            elem = self.builder.load(elem_ptr, name="fill_load")
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name="promote_fill_elem")
            pred = self.builder.call(func, [elem], name="fill_pred")
            pred_bool = self._convert_to_bool(pred)
            
            fill_pred_block = self.func.append_basic_block(name="fill_pred")
            fill_skip_block = self.func.append_basic_block(name="fill_skip")
            self.builder.cbranch(pred_bool, fill_pred_block, fill_skip_block)
            
            self.builder.position_at_end(fill_pred_block)
            curr_count = self.builder.load(count, name="fill_curr_count")
            result_ptr = self.builder.gep(temp_result, [ir.Constant(self.int_type, 0), curr_count], name="fill_result_ptr")
            self.builder.store(elem, result_ptr)
            next_count = self.builder.add(curr_count, ir.Constant(self.int_type, 1), name="fill_next_count")
            self.builder.store(next_count, count)
            self.builder.branch(fill_skip_block)
            
            self.builder.position_at_end(fill_skip_block)
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="fill_next_index")
            self.builder.store(next_index, index)
            self.builder.branch(fill_loop_block)
            
            self.builder.position_at_end(fill_exit_block)
            
            # Create final result struct
            struct_type = ir.LiteralStructType([ir.PointerType(self.float_type), self.int_type])
            result_struct = self.builder.alloca(struct_type, name="filter_struct")
            count_ptr = self.builder.gep(result_struct, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="filter_count_ptr")
            self.builder.store(final_count, count_ptr)
            
            # Allocate final array with exact size
            final_result = self.builder.alloca(result_type, name="final_filter_result")
            array_ptr = self.builder.gep(final_result, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="final_array_ptr")
            struct_array_ptr = self.builder.gep(result_struct, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="struct_array_ptr")
            self.builder.store(array_ptr, struct_array_ptr)
            
            # Copy valid elements to final array
            copy_loop_block = self.func.append_basic_block(name="filter_copy_loop")
            copy_body_block = self.func.append_basic_block(name="filter_copy_body")
            copy_exit_block = self.func.append_basic_block(name="filter_copy_exit")
            
            copy_index = self.builder.alloca(self.int_type, name="copy_index")
            self.builder.store(ir.Constant(self.int_type, 0), copy_index)
            self.builder.branch(copy_loop_block)
            
            self.builder.position_at_end(copy_loop_block)
            curr_copy_index = self.builder.load(copy_index, name="curr_copy_index")
            copy_cond = self.builder.icmp_signed("<", curr_copy_index, final_count, name="copy_cond")
            self.builder.cbranch(copy_cond, copy_body_block, copy_exit_block)
            
            self.builder.position_at_end(copy_body_block)
            src_ptr = self.builder.gep(temp_result, [ir.Constant(self.int_type, 0), curr_copy_index], name="src_elem")
            dst_ptr = self.builder.gep(final_result, [ir.Constant(self.int_type, 0), curr_copy_index], name="dst_elem")
            elem = self.builder.load(src_ptr, name="copy_load")
            self.builder.store(elem, dst_ptr)
            next_copy_index = self.builder.add(curr_copy_index, ir.Constant(self.int_type, 1), name="next_copy_index")
            self.builder.store(next_copy_index, copy_index)
            self.builder.branch(copy_loop_block)
            
            self.builder.position_at_end(copy_exit_block)
            return result_struct
        elif name == "list_reduce":
            acc = self.builder.alloca(self.float_type, name="reduce_acc")
            if extra_arg is not None:
                init_val = extra_arg
                if isinstance(init_val.type, ir.IntType):
                    init_val = self.builder.sitofp(init_val, self.float_type, name="promote_init")
                self.builder.store(init_val, acc)
                start_index = 0
            else:
                if list_size == 0:
                    return ir.Constant(self.float_type, 0.0)
                elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="reduce_init")
                init_val = self.builder.load(elem_ptr, name="reduce_init_load")
                if isinstance(init_val.type, ir.IntType):
                    init_val = self.builder.sitofp(init_val, self.float_type, name="promote_init")
                self.builder.store(init_val, acc)
                start_index = 1
            index = self.builder.alloca(self.int_type, name="reduce_index")
            self.builder.store(ir.Constant(self.int_type, start_index), index)
            
            loop_block = self.func.append_basic_block(name="reduce_loop")
            body_block = self.func.append_basic_block(name="reduce_body")
            exit_block = self.func.append_basic_block(name="reduce_exit")
            
            self.builder.branch(loop_block)
            self.builder.position_at_end(loop_block)
            
            curr_index = self.builder.load(index, name="curr_index")
            cond = self.builder.icmp_signed("<", curr_index, ir.Constant(self.int_type, list_size), name="reduce_cond")
            self.builder.cbranch(cond, body_block, exit_block)
            
            self.builder.position_at_end(body_block)
            elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), curr_index], name="reduce_elem")
            elem = self.builder.load(elem_ptr, name="reduce_load")
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name="promote_elem")
            acc_val = self.builder.load(acc, name="acc_load")
            result = self.builder.call(func, [acc_val, elem], name="reduce_call")
            self.builder.store(result, acc)
            
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_index")
            self.builder.store(next_index, index)
            self.builder.branch(loop_block)
            
            self.builder.position_at_end(exit_block)
            return self.builder.load(acc, name="reduce_result")
        
        elif name == "list_group_by":
            # Define a struct type to hold false_list and true_list
            group_type = ir.LiteralStructType([
                ir.ArrayType(self.float_type, list_size),  # false_list
                ir.ArrayType(self.float_type, list_size),  # true_list
                self.int_type,  # false_count
                self.int_type   # true_count
            ])
            result_alloca = self.builder.alloca(group_type, name="group_result")
            
            # Initialize arrays and counters
            false_list_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name="false_list")
            true_list_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name="true_list")
            false_count_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 2)], name="false_count")
            true_count_ptr = self.builder.gep(result_alloca, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 3)], name="true_count")
            
            self.builder.store(ir.Constant(self.int_type, 0), false_count_ptr)
            self.builder.store(ir.Constant(self.int_type, 0), true_count_ptr)
            
            # Initialize arrays to 0.0 to avoid garbage values
            for i in range(list_size):
                false_elem_ptr = self.builder.gep(false_list_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"false_init_{i}")
                true_elem_ptr = self.builder.gep(true_list_ptr, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, i)], name=f"true_init_{i}")
                self.builder.store(ir.Constant(self.float_type, 0.0), false_elem_ptr)
                self.builder.store(ir.Constant(self.float_type, 0.0), true_elem_ptr)
            
            index = self.builder.alloca(self.int_type, name="group_index")
            self.builder.store(ir.Constant(self.int_type, 0), index)
            
            loop_block = self.func.append_basic_block(name="group_loop")
            body_block = self.func.append_basic_block(name="group_body")
            exit_block = self.func.append_basic_block(name="group_exit")
            
            self.builder.branch(loop_block)
            self.builder.position_at_end(loop_block)
            
            curr_index = self.builder.load(index, name="curr_index")
            cond = self.builder.icmp_signed("<", curr_index, ir.Constant(self.int_type, list_size), name="group_cond")
            self.builder.cbranch(cond, body_block, exit_block)
            
            self.builder.position_at_end(body_block)
            elem_ptr = self.builder.gep(list_ptr, [ir.Constant(self.int_type, 0), curr_index], name="group_elem")
            elem = self.builder.load(elem_ptr, name="group_load")
            # Ensure elem is double
            if isinstance(elem.type, ir.IntType):
                elem = self.builder.sitofp(elem, self.float_type, name="promote_group_elem")
            self._dprint(f"Calling group function {func} with arg: {elem} (type: {elem.type})")
            key = self.builder.call(func, [elem], name="group_call")
            key_bool = self._convert_to_bool(key)
            
            true_block = self.func.append_basic_block(name="group_true")
            false_block = self.func.append_basic_block(name="group_false")
            continue_block = self.func.append_basic_block(name="group_continue")
            self.builder.cbranch(key_bool, true_block, false_block)
            
            self.builder.position_at_end(true_block)
            true_curr_count = self.builder.load(true_count_ptr, name="true_curr_count")
            true_ptr = self.builder.gep(true_list_ptr, [ir.Constant(self.int_type, 0), true_curr_count], name="true_ptr")
            self.builder.store(elem, true_ptr)
            true_next_count = self.builder.add(true_curr_count, ir.Constant(self.int_type, 1), name="true_next_count")
            self.builder.store(true_next_count, true_count_ptr)
            self.builder.branch(continue_block)
            
            self.builder.position_at_end(false_block)
            false_curr_count = self.builder.load(false_count_ptr, name="false_curr_count")
            false_ptr = self.builder.gep(false_list_ptr, [ir.Constant(self.int_type, 0), false_curr_count], name="false_ptr")
            self.builder.store(elem, false_ptr)
            false_next_count = self.builder.add(false_curr_count, ir.Constant(self.int_type, 1), name="false_next_count")
            self.builder.store(false_next_count, false_count_ptr)
            self.builder.branch(continue_block)
            
            self.builder.position_at_end(continue_block)
            next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name="next_index")
            self.builder.store(next_index, index)
            self.builder.branch(loop_block)
            
            self.builder.position_at_end(exit_block)
            return result_alloca

    def _generate_call(self, node):
        self._dprint(f"Generating call: {node}")
        if isinstance(node.callee, AtomicNode) and node.callee.type == "identifier":
            callee_name = node.callee.value
            if callee_name == "print":
                return self._handle_print_call(node)
            if callee_name in ["list_map", "list_filter", "list_reduce", "list_group_by"]:
                args = [self._generate_expr(arg) for arg in node.args]
                if callee_name == "list_reduce" and len(node.args) == 3:
                    return self._generate_builtin_function(callee_name, args[0], args[1], args[2])
                elif len(args) == 2:
                    return self._generate_builtin_function(callee_name, args[0], args[1])
                else:
                    raise Exception(f"{callee_name} expects 2 or 3 arguments")
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
            param_type = func.type.pointee.args[i] if i < len(func.type.pointee.args) else self.float_type
            if isinstance(param_type, ir.DoubleType) and isinstance(arg_val.type, ir.IntType) and arg_val.type.width > 1:
                arg_val = self.builder.sitofp(arg_val, self.float_type, name="promote_arg")
            args.append(arg_val)
        return self.builder.call(func, args, name="calltmp")

    def _handle_print_call(self, node):
        self._dprint(f"Generating print call with args: {node.args}")
        printf = self.module.globals.get('printf')
        if not printf:
            voidptr_ty = self.string_type
            printf_ty = ir.FunctionType(self.int_type, [voidptr_ty], var_arg=True)
            printf = ir.Function(self.module, printf_ty, name="printf")

        for i, arg_node in enumerate(node.args):
            arg = self._generate_expr(arg_node)
            if isinstance(arg_node, AtomicNode) and arg_node.type == "string" and arg_node.value.strip() == "":
                continue

            if isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.ArrayType):
                # Handle regular list printing
                count = ir.Constant(self.int_type, arg.type.pointee.count)
                if isinstance(arg_node, BinaryNode) and arg_node.operator == "INDEX":
                    struct_ptr = self._generate_expr(arg_node.left)
                    struct_type = struct_ptr.type.pointee
                    if isinstance(struct_type, ir.LiteralStructType) and len(struct_type.elements) == 4:
                        index_val = 1 if arg_node.right.value else 0
                        count_index = ir.Constant(self.int_type, 2 + index_val)
                        count_ptr = self.builder.gep(struct_ptr, [ir.Constant(self.int_type, 0), count_index], name=f"count_ptr_{i}")
                        count = self.builder.load(count_ptr, name=f"count_{i}")

                self.builder.call(printf, [self._create_global_string("[")], name=f"print_open_bracket_{i}")
                index_var = self.builder.alloca(self.int_type, name=f"print_index_{i}")
                self.builder.store(ir.Constant(self.int_type, 0), index_var)
                loop_block = self.func.append_basic_block(name=f"print_loop_{i}")
                body_block = self.func.append_basic_block(name=f"print_body_{i}")
                exit_block = self.func.append_basic_block(name=f"print_exit_{i}")

                self.builder.branch(loop_block)
                self.builder.position_at_end(loop_block)
                curr_index = self.builder.load(index_var, name=f"curr_print_index_{i}")
                cond = self.builder.icmp_signed("<", curr_index, count, name=f"print_cond_{i}")
                self.builder.cbranch(cond, body_block, exit_block)

                self.builder.position_at_end(body_block)
                not_first_block = self.func.append_basic_block(name=f"print_not_first_{i}")
                first_block = self.func.append_basic_block(name=f"print_first_{i}")
                continue_block = self.func.append_basic_block(name=f"print_continue_{i}")
                is_not_first = self.builder.icmp_signed("!=", curr_index, ir.Constant(self.int_type, 0), name=f"is_not_first_{i}")
                self.builder.cbranch(is_not_first, not_first_block, first_block)

                self.builder.position_at_end(not_first_block)
                comma_str = self._create_global_string(", ")
                self.builder.call(printf, [comma_str], name=f"print_comma_{i}")
                self.builder.branch(continue_block)

                self.builder.position_at_end(first_block)
                self.builder.branch(continue_block)

                self.builder.position_at_end(continue_block)
                elem_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), curr_index], name=f"list_elem_{i}")
                elem = self.builder.load(elem_ptr, name=f"load_elem_{i}")
                int_val = self.builder.fptosi(elem, self.int_type, name=f"to_int_{i}")
                float_val = self.builder.sitofp(int_val, self.float_type, name=f"to_float_{i}")
                is_int = self.builder.fcmp_ordered("==", elem, float_val, name=f"is_int_{i}")
                int_block = self.func.append_basic_block(name=f"print_int_{i}")
                float_block = self.func.append_basic_block(name=f"print_float_{i}")
                print_continue_block = self.func.append_basic_block(name=f"print_continue_elem_{i}")
                self.builder.cbranch(is_int, int_block, float_block)

                self.builder.position_at_end(int_block)
                int_fmt = self._create_global_string("%d")
                self.builder.call(printf, [int_fmt, int_val], name=f"print_elem_int_{i}")
                self.builder.branch(print_continue_block)

                self.builder.position_at_end(float_block)
                float_fmt = self._create_global_string("%.6f")
                self.builder.call(printf, [float_fmt, elem], name=f"print_elem_float_{i}")
                self.builder.branch(print_continue_block)

                self.builder.position_at_end(print_continue_block)
                next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"next_print_index_{i}")
                self.builder.store(next_index, index_var)
                self.builder.branch(loop_block)

                self.builder.position_at_end(exit_block)
                self.builder.call(printf, [self._create_global_string("]")], name=f"print_close_bracket_{i}")

            elif isinstance(arg.type, ir.PointerType) and isinstance(arg.type.pointee, ir.LiteralStructType):
                struct_type = arg.type.pointee
                if len(struct_type.elements) == 4:  # list_group_by struct
                    self.builder.call(printf, [self._create_global_string("#{false: ")], name=f"print_group_open_{i}")
                    false_list_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name=f"false_list_{i}")
                    false_count_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 2)], name=f"false_count_{i}")
                    false_count = self.builder.load(false_count_ptr, name=f"false_count_load_{i}")
                    
                    index_var = self.builder.alloca(self.int_type, name=f"false_print_index_{i}")
                    self.builder.store(ir.Constant(self.int_type, 0), index_var)
                    loop_block = self.func.append_basic_block(name=f"false_print_loop_{i}")
                    body_block = self.func.append_basic_block(name=f"false_print_body_{i}")
                    exit_block = self.func.append_basic_block(name=f"false_print_exit_{i}")

                    self.builder.call(printf, [self._create_global_string("[")], name=f"false_print_open_{i}")
                    self.builder.branch(loop_block)
                    self.builder.position_at_end(loop_block)
                    curr_index = self.builder.load(index_var, name=f"false_curr_index_{i}")
                    cond = self.builder.icmp_signed("<", curr_index, false_count, name=f"false_print_cond_{i}")
                    self.builder.cbranch(cond, body_block, exit_block)

                    self.builder.position_at_end(body_block)
                    not_first_block = self.func.append_basic_block(name=f"false_not_first_{i}")
                    first_block = self.func.append_basic_block(name=f"false_first_{i}")
                    continue_block = self.func.append_basic_block(name=f"false_continue_{i}")
                    is_not_first = self.builder.icmp_signed("!=", curr_index, ir.Constant(self.int_type, 0), name=f"false_is_not_first_{i}")
                    self.builder.cbranch(is_not_first, not_first_block, first_block)

                    self.builder.position_at_end(not_first_block)
                    self.builder.call(printf, [self._create_global_string(", ")], name=f"false_print_comma_{i}")
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(first_block)
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(continue_block)
                    elem_ptr = self.builder.gep(false_list_ptr, [ir.Constant(self.int_type, 0), curr_index], name=f"false_elem_{i}")
                    elem = self.builder.load(elem_ptr, name=f"false_load_elem_{i}")
                    int_val = self.builder.fptosi(elem, self.int_type, name=f"false_to_int_{i}")
                    float_val = self.builder.sitofp(int_val, self.float_type, name=f"false_to_float_{i}")
                    is_int = self.builder.fcmp_ordered("==", elem, float_val, name=f"false_is_int_{i}")
                    int_block = self.func.append_basic_block(name=f"false_print_int_{i}")
                    float_block = self.func.append_basic_block(name=f"false_print_float_{i}")
                    print_continue_block = self.func.append_basic_block(name=f"false_print_continue_elem_{i}")
                    self.builder.cbranch(is_int, int_block, float_block)

                    self.builder.position_at_end(int_block)
                    self.builder.call(printf, [self._create_global_string("%d"), int_val], name=f"false_print_elem_int_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(float_block)
                    self.builder.call(printf, [self._create_global_string("%.6f"), elem], name=f"false_print_elem_float_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(print_continue_block)
                    next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"false_next_index_{i}")
                    self.builder.store(next_index, index_var)
                    self.builder.branch(loop_block)

                    self.builder.position_at_end(exit_block)
                    self.builder.call(printf, [self._create_global_string("], true: ")], name=f"false_print_close_{i}")

                    true_list_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name=f"true_list_{i}")
                    true_count_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 3)], name=f"true_count_{i}")
                    true_count = self.builder.load(true_count_ptr, name=f"true_count_load_{i}")
                    
                    index_var = self.builder.alloca(self.int_type, name=f"true_print_index_{i}")
                    self.builder.store(ir.Constant(self.int_type, 0), index_var)
                    loop_block = self.func.append_basic_block(name=f"true_print_loop_{i}")
                    body_block = self.func.append_basic_block(name=f"true_print_body_{i}")
                    exit_block = self.func.append_basic_block(name=f"true_print_exit_{i}")

                    self.builder.call(printf, [self._create_global_string("[")], name=f"true_print_open_{i}")
                    self.builder.branch(loop_block)
                    self.builder.position_at_end(loop_block)
                    curr_index = self.builder.load(index_var, name=f"true_curr_index_{i}")
                    cond = self.builder.icmp_signed("<", curr_index, true_count, name=f"true_print_cond_{i}")
                    self.builder.cbranch(cond, body_block, exit_block)

                    self.builder.position_at_end(body_block)
                    not_first_block = self.func.append_basic_block(name=f"true_not_first_{i}")
                    first_block = self.func.append_basic_block(name=f"true_first_{i}")
                    continue_block = self.func.append_basic_block(name=f"true_continue_{i}")
                    is_not_first = self.builder.icmp_signed("!=", curr_index, ir.Constant(self.int_type, 0), name=f"true_is_not_first_{i}")
                    self.builder.cbranch(is_not_first, not_first_block, first_block)

                    self.builder.position_at_end(not_first_block)
                    self.builder.call(printf, [self._create_global_string(", ")], name=f"true_print_comma_{i}")
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(first_block)
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(continue_block)
                    elem_ptr = self.builder.gep(true_list_ptr, [ir.Constant(self.int_type, 0), curr_index], name=f"true_elem_{i}")
                    elem = self.builder.load(elem_ptr, name=f"true_load_elem_{i}")
                    int_val = self.builder.fptosi(elem, self.int_type, name=f"true_to_int_{i}")
                    float_val = self.builder.sitofp(int_val, self.float_type, name=f"true_to_float_{i}")
                    is_int = self.builder.fcmp_ordered("==", elem, float_val, name=f"true_is_int_{i}")
                    int_block = self.func.append_basic_block(name=f"true_print_int_{i}")
                    float_block = self.func.append_basic_block(name=f"true_print_float_{i}")
                    print_continue_block = self.func.append_basic_block(name=f"true_print_continue_elem_{i}")
                    self.builder.cbranch(is_int, int_block, float_block)

                    self.builder.position_at_end(int_block)
                    self.builder.call(printf, [self._create_global_string("%d"), int_val], name=f"true_print_elem_int_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(float_block)
                    self.builder.call(printf, [self._create_global_string("%.6f"), elem], name=f"true_print_elem_float_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(print_continue_block)
                    next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"true_next_index_{i}")
                    self.builder.store(next_index, index_var)
                    self.builder.branch(loop_block)

                    self.builder.position_at_end(exit_block)
                    self.builder.call(printf, [self._create_global_string("]}")], name=f"true_print_close_{i}")
                
                elif len(struct_type.elements) == 2:  # list_filter or range struct
                    array_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 0)], name=f"array_ptr_{i}")
                    array = self.builder.load(array_ptr, name=f"array_{i}")
                    count_ptr = self.builder.gep(arg, [ir.Constant(self.int_type, 0), ir.Constant(self.int_type, 1)], name=f"count_ptr_{i}")
                    count = self.builder.load(count_ptr, name=f"count_{i}")

                    index_var = self.builder.alloca(self.int_type, name=f"print_index_{i}")
                    self.builder.store(ir.Constant(self.int_type, 0), index_var)
                    loop_block = self.func.append_basic_block(name=f"print_loop_{i}")
                    body_block = self.func.append_basic_block(name=f"print_body_{i}")
                    exit_block = self.func.append_basic_block(name=f"print_exit_{i}")

                    self.builder.call(printf, [self._create_global_string("[")], name=f"print_open_{i}")
                    self.builder.branch(loop_block)
                    self.builder.position_at_end(loop_block)
                    curr_index = self.builder.load(index_var, name=f"curr_index_{i}")
                    cond = self.builder.icmp_signed("<", curr_index, count, name=f"print_cond_{i}")
                    self.builder.cbranch(cond, body_block, exit_block)

                    self.builder.position_at_end(body_block)
                    not_first_block = self.func.append_basic_block(name=f"not_first_{i}")
                    first_block = self.func.append_basic_block(name=f"first_{i}")
                    continue_block = self.func.append_basic_block(name=f"continue_{i}")
                    is_not_first = self.builder.icmp_signed("!=", curr_index, ir.Constant(self.int_type, 0), name=f"is_not_first_{i}")
                    self.builder.cbranch(is_not_first, not_first_block, first_block)

                    self.builder.position_at_end(not_first_block)
                    self.builder.call(printf, [self._create_global_string(", ")], name=f"print_comma_{i}")
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(first_block)
                    self.builder.branch(continue_block)

                    self.builder.position_at_end(continue_block)
                    elem_ptr = self.builder.gep(array, [curr_index], name=f"elem_{i}")
                    elem = self.builder.load(elem_ptr, name=f"load_elem_{i}")
                    int_val = self.builder.fptosi(elem, self.int_type, name=f"to_int_{i}")
                    float_val = self.builder.sitofp(int_val, self.float_type, name=f"to_float_{i}")
                    is_int = self.builder.fcmp_ordered("==", elem, float_val, name=f"is_int_{i}")
                    int_block = self.func.append_basic_block(name=f"print_int_{i}")
                    float_block = self.func.append_basic_block(name=f"print_float_{i}")
                    print_continue_block = self.func.append_basic_block(name=f"print_continue_elem_{i}")
                    self.builder.cbranch(is_int, int_block, float_block)

                    self.builder.position_at_end(int_block)
                    self.builder.call(printf, [self._create_global_string("%d"), int_val], name=f"print_elem_int_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(float_block)
                    self.builder.call(printf, [self._create_global_string("%.6f"), elem], name=f"print_elem_float_{i}")
                    self.builder.branch(print_continue_block)

                    self.builder.position_at_end(print_continue_block)
                    next_index = self.builder.add(curr_index, ir.Constant(self.int_type, 1), name=f"next_index_{i}")
                    self.builder.store(next_index, index_var)
                    self.builder.branch(loop_block)

                    self.builder.position_at_end(exit_block)
                    self.builder.call(printf, [self._create_global_string("]")], name=f"print_close_{i}")

            elif isinstance(arg.type, ir.DoubleType):
                int_val = self.builder.fptosi(arg, self.int_type, name=f"to_int_{i}")
                float_val = self.builder.sitofp(int_val, self.float_type, name=f"to_float_{i}")
                is_int = self.builder.fcmp_ordered("==", arg, float_val, name=f"is_int_{i}")
                int_block = self.func.append_basic_block(name=f"print_int_{i}")
                float_block = self.func.append_basic_block(name=f"print_float_{i}")
                continue_block = self.func.append_basic_block(name=f"print_continue_{i}")
                self.builder.cbranch(is_int, int_block, float_block)

                self.builder.position_at_end(int_block)
                int_fmt = self._create_global_string("%d")
                self.builder.call(printf, [int_fmt, int_val], name=f"print_elem_int_{i}")
                self.builder.branch(continue_block)

                self.builder.position_at_end(float_block)
                float_fmt = self._create_global_string("%.6f")
                self.builder.call(printf, [float_fmt, arg], name=f"print_elem_float_{i}")
                self.builder.branch(continue_block)

                self.builder.position_at_end(continue_block)

            elif isinstance(arg.type, ir.IntType) and arg.type.width == 1:
                true_str = self._create_global_string("true")
                false_str = self._create_global_string("false")
                result_ptr = self.builder.select(arg, true_str, false_str, name="bool_str")
                fmt_ptr = self._create_global_string("%s")
                self.builder.call(printf, [fmt_ptr, result_ptr], name=f"calltmp_{i}")

            elif isinstance(arg.type, ir.IntType):
                fmt_ptr = self._create_global_string("%d")
                self.builder.call(printf, [fmt_ptr, arg], name=f"calltmp_{i}")

            elif isinstance(arg.type, ir.PointerType):
                fmt_ptr = self._create_global_string("%s")
                self.builder.call(printf, [fmt_ptr, arg], name=f"calltmp_{i}")

            else:
                raise Exception(f"Unsupported print argument type: {arg.type}")

            if i < len(node.args) - 1:
                space_str = self._create_global_string(" ")
                self.builder.call(printf, [space_str], name=f"print_space_{i}")

        newline_str = self._create_global_string("\n")
        return self.builder.call(printf, [newline_str], name="calltmp_newline")