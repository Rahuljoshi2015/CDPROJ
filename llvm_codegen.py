from llvmlite import ir, binding
from myast import AtomicNode, BinaryNode, BlockNode, IfNode, LambdaNode, ListNode, MapNode, Node, ProgramNode, SliceNode, TupleNode, UnaryNode, CallNode  # Adjust import to match your project

class LLVMCodeGenerator:
    def __init__(self):
        self.module = ir.Module(name="main")
        self.builder = None
        self.func = None
        self.symbol_table = {}

        # Basic types
        self.int_type = ir.IntType(32)
        self.bool_type = ir.IntType(1)
        self.void_type = ir.VoidType()
        self.float_type = ir.DoubleType()

    def generate_code(self, node):
        if isinstance(node, ProgramNode):
            self._generate_program(node)
        else:
            raise Exception(f"Unhandled root node type: {type(node)}")
        return str(self.module)

    def _generate_program(self, program_node):
        # Create main function: int main()
        func_type = ir.FunctionType(self.int_type, [])
        self.func = ir.Function(self.module, func_type, name="main")
        block = self.func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Generate code for each expression
        for expr in program_node.expressions:
            self._generate_expr(expr)

        self.builder.ret(ir.Constant(self.int_type, 0))

    
    def _generate_expr(self, node):
     if isinstance(node, AtomicNode):
        return self._generate_atomic(node)
     elif isinstance(node, BinaryNode):
        return self._generate_binary(node)
     elif isinstance(node, CallNode):
        return self._generate_call(node)
     else:
        raise Exception(f"Unknown expression node: {type(node)}")


    
    def _generate_atomic(self, node):
     if node.type == "number":
        return ir.Constant(self.float_type, node.value)
     elif node.type == "bool":
        return ir.Constant(self.bool_type, int(node.value))
     elif node.type == "identifier":
        if node.value in self.symbol_table:
            return self.builder.load(self.symbol_table[node.value], name=node.value)
        else:
            raise Exception(f"Undefined variable: {node.value}")
     elif node.type == "string":
        return self.builder.global_string_ptr(node.value)

     else:
        raise Exception(f"Unsupported atomic type: {node.type}")


    def _generate_binary(self, node):
        left = self._generate_expr(node.left)
        right = self._generate_expr(node.right)

        if node.op == "PLUS":
            return self.builder.fadd(left, right, name="addtmp")
        elif node.op == "MINUS":
            return self.builder.fsub(left, right, name="subtmp")
        elif node.op == "MULTIPLY":
            return self.builder.fmul(left, right, name="multmp")
        elif node.op == "DIVIDE":
            return self.builder.fdiv(left, right, name="divtmp")
        else:
            raise Exception(f"Unsupported binary operator: {node.op}")
        
    def _generate_call(self, node):
    # Determine the name of the callee
        if isinstance(node.callee, str):
            callee_name = node.callee
        elif isinstance(node.callee, AtomicNode) and node.callee.type == "identifier":
            callee_name = node.callee.value
        else:
            raise Exception(f"Unsupported callee type: {type(node.callee)}")

        # Handle the 'print' function
        if callee_name == "print":
            printf = self.module.globals.get('printf')
            if not printf:
                voidptr_ty = ir.IntType(8).as_pointer()
                printf_ty = ir.FunctionType(ir.IntType(32), [voidptr_ty], var_arg=True)
                printf = ir.Function(self.module, printf_ty, name="printf")

            llvm_args = []
            fmt_parts = []

            for arg_node in node.args:
                arg = self._generate_expr(arg_node)
                if isinstance(arg_node, AtomicNode) and arg_node.type == "string":
                    fmt_parts.append(arg_node.value)
                elif isinstance(arg.type, ir.DoubleType):
                    fmt_parts.append("%f")
                    llvm_args.append(arg)
                elif isinstance(arg.type, ir.IntType):
                    fmt_parts.append("%d")
                    llvm_args.append(arg)
                else:
                    raise Exception("Unsupported print argument type")

            fmt = " ".join(fmt_parts) + "\n\0"
            fmt_arg = self.builder.global_string(fmt, name="fmt")
            return self.builder.call(printf, [fmt_arg] + llvm_args, name="calltmp")

        # If the function is not recognized
        else:
            raise Exception(f"Unknown function: {callee_name}")




