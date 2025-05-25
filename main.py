import sys
from io import StringIO

from lexer import Lexer
from parser import Parser
from llvm_codegen import LLVMCodeGenerator

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <source_file>")
        return

    filename = sys.argv[1]

    try:
        with open(filename, "r", encoding="utf-8") as f:
            source_code = f.read()

        lexer = Lexer(StringIO(source_code), debug=False)
        parser = Parser(lexer, debug=False)
        ast_tree = parser.parse()
        #print("AST:")
        #print(ast_tree)
        codegen = LLVMCodeGenerator()
        llvm_ir = codegen.generate_code(ast_tree)
        

        # ✅ Write LLVM IR to file in UTF-8
        with open("output.ll", "w", encoding="utf-8") as out_file:
            out_file.write(llvm_ir)

        print("LLVM IR written to output.ll")

    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    main()


