from io import StringIO
import sys

from lexer import Lexer
from parser import Parser
from myast import AtomicNode, BinaryNode, BlockNode, IfNode, LambdaNode, ListNode, MapNode, Node, ProgramNode, SliceNode, TupleNode, UnaryNode, IdentifierNode

def main():
    # Check if filename provided as argument
    if len(sys.argv) < 2:
        print("Usage: python ast_generator.py <source_file>")
        return

    filename = sys.argv[1]

    try:
        with open(filename, "r", encoding="utf-8") as f:
            source_code = f.read()

        # Initialize Lexer and Parser like in interpreter
        lexer = Lexer(StringIO(source_code), debug=False)
        parser = Parser(lexer, debug=False)
        ast_tree: ProgramNode = parser.parse()

        # Print the formatted AST
        print("Generated AST:")
        print(ast_tree.formatted_str())

    except Exception as e:
        print(f"Parsing failed: {e}")

if __name__ == "__main__":
    main()