# test_lexer.py

from lexer import Lexer
from io import StringIO

def test_lexer():
    source_code = '''
print("Value", "\t", "Is correct")
result = 1 + 2 * 3
print(result, "\t", result == 7)
result = 3 * 2 + 1
print(result, "\t", result == 7)
result = 7 - 5 + 2
print(result, "\t", result == 4)
result = (10 + 5) * 3 - 10 / 5 - 5 + 4
print(result, "\t", result == 42)
result = 5 * 10 + 10 ^ 2 / 20
print(result, "\t", result == 55)
result = -1 - -(2 - -3)
print(result, "\t", result == 4)
result = true && false || true && (true && false || false)
print(result, "\t", result == false)
    '''

    source = StringIO(source_code)

    # Set debug=False if your Lexer has a debug flag to suppress internal printing
    lexer = Lexer(source, debug=False)

    while not lexer.is_done():
        token = lexer.next_token()
        print(token)  # Only this should print

if __name__ == "__main__":
    test_lexer()
