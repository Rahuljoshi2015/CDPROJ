class Node():
    """
    A base node in the abstract syntax tree.
    """
    def __init__(self, name):
        """
        Initialize a node with a name.
        """
        self.name = name

    def __str__(self):
        return self.formatted_str()
    
    def raw_str(self):
        """
        Returns the raw value without any formatting.
        For example, a string node will return **ONLY** the string **with** quotes.
        """
        return self.formatted_str()
    
    def formatted_str(self):
        """
        Returns the formatted value.
        For example, a string node will return the string **with** quotes and the type, e.g. `string('hello')`.
        """
        return str(self.name)

class ProgramNode(Node):
    """
    A program node in the abstract syntax tree.
    """
    def __init__(self, expressions: list[Node]):
        """
        Initialize a program node with a list of expressions.
        """
        super().__init__("Program")
        self.expressions = expressions
    def formatted_str(self):
        lines = ["Program:"]
        for expr in self.expressions:
            # indent each expression by two spaces, recursively formatted
            formatted_expr = expr.formatted_str().replace('\n', '\n  ')
            lines.append("  " + formatted_expr)
        return "\n".join(lines)



class AtomicNode(Node):
    """
    An atomic expression node in the abstract syntax tree.
    """
    def __init__(self, type: str, value):
        """
        Initialize an atomic expression node with a value.
        """
        super().__init__("Atomic")
        self.type = type
        self.value = value


    def raw_str(self):
        """
        Returns the raw value without any formatting.
        For example, a string node will return **ONLY** the string **with** quotes.
        """
        if self.type == "string":
            return f"'{self.value}'"
        return str(self.value)
    
    def formatted_str(self):
        """
        Returns the formatted value.
        For example, a string node will return the string **with** quotes and the type, e.g. `string('hello')`.
        """
        if self.type == "string":
            return f"'{self.value}'"
        return f"{self.type}({self.value})"

class BlockNode(Node):
    """
    A block node in the abstract syntax tree.
    """
    def __init__(self, expressions: list[Node]):
        """
        Initialize a block node with a list of expressions.
        """
        super().__init__("Block")
        self.expressions = expressions

    def formatted_str(self):
        return '{' + '; '.join(map(str, self.expressions)) + '}'

class LazyRangeNode(Node):
    """
    A node for lazy range expressions (e.g., 1..10).
    """
    def __init__(self, start: Node, end: Node):
        super().__init__("LazyRange")
        self.start = start
        self.end = end

    def formatted_str(self):
        return f"{self.start}..{self.end}"
    
class TupleNode(Node):
    """
    A tuple node in the abstract syntax tree.
    """
    def __init__(self, elements: list[Node]):
        """
        Initialize a tuple node with a list of elements.
        """
        super().__init__("Tuple")
        self.elements = elements

    def formatted_str(self):
        return '(' + ", ".join(map(str, self.elements)) + ')'

class ListNode(Node):
    """
    A list node in the abstract syntax tree.
    """
    def __init__(self, elements: list[Node]):
        """
        Initialize a list node with a list of elements.
        """
        super().__init__("List")
        self.elements = elements

    def formatted_str(self):
        return '[' + ", ".join(map(str, self.elements)) + ']'

class SliceNode(Node):
    """
    A slice node in the abstract syntax tree.
    """
    def __init__(self, start: int, end: int, step: int | None):
        """
        Initialize a slice node with a start, end and step.
        """
        super().__init__("Slice")
        self.start = start
        self.end = end
        self.step = step

    def formatted_str(self):
        step = f":{self.step}" if self.step is not None else ""
        return f"<slice {self.start}:{self.end}{step}>"

class MapNode(Node):
    """
    A hash map node in the abstract syntax tree.
    """
    def __init__(self, pairs: dict[Node, Node]):
        """
        Initialize a map node with a list of elements.
        """
        super().__init__("Map")
        self.pairs = pairs

    def formatted_str(self):
        return '#{' + ', '.join(map(lambda t: f"{t[0]}: {t[1]}", self.pairs.items())) + '}'
    
class ForNode(Node):
        """
        A for loop node in the abstract syntax tree.
        """
        def __init__(self, var_name: str, iterable: Node, body: Node):
            """
            Initialize a for loop node with a variable name, iterable, and body.
            """
            super().__init__("For")
            self.var_name = var_name
            self.iterable = iterable
            self.body = body

        def formatted_str(self):
            return f"for {self.var_name} in {self.iterable} {self.body}"

class UnaryNode(Node):
    """
    A unary node in the abstract syntax tree.
    """
    def __init__(self, operator: str, rhs: Node):
        """
        Initialize a unary expression node with an operator and a right
        expression.
        """
        super().__init__("Unary")
        self.operator = operator
        self.rhs = rhs

    def formatted_str(self):
        return f"{self.operator}({self.rhs})"

class BinaryNode(Node):
    """
    A binary expression node in the abstract syntax tree.
    """
    def __init__(self, operator: str, left: Node, right: Node):
        """
        Initialize a binary expression node with an operator, left and right
        expressions.
        """
        super().__init__("Binary")
        self.operator = operator
        self.left = left
        self.right = right

    def formatted_str(self):
        return f"({self.left} {self.operator} {self.right})"

class LambdaNode(Node):
    """
    A lambda function node in the abstract syntax tree.
    """
    def __init__(self, params: list[str], body: Node):
        """
        Initialize a lambda function node with a list of parameters and a body.
        """
        super().__init__("Lambda")
        self.params = params
        self.body = body

    def formatted_str(self):
        return f"({self.params} => {self.body})"

class IfNode(Node):
    """
    An if node in the abstract syntax tree.
    Contains a condition and body togehter with an optional if else statements and else body.
    """
    def __init__(self, condition: Node, ifBody: Node, elseIfs: list[tuple[Node, Node]] = [], elseBody: Node = None):
        """
        Initialize an if node with a condition, if body and an optional else body and else ifs.
        """
        super().__init__("If")
        self.condition = condition
        self.ifBody = ifBody
        self.elseBody = elseBody
        self.elseIfs = elseIfs
        
class CallNode(Node):
    """
    A function call node in the abstract syntax tree.
    """
    def __init__(self, callee: str, args: list[Node]):
        """
        Initialize a call node with the callee name and a list of argument nodes.
        """
        super().__init__("Call")
        self.callee = callee
        self.args = args

    def formatted_str(self):
        return f"{self.callee}({', '.join(map(str, self.args))})"
