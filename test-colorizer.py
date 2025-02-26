#!/usr/bin/env python3
"""
Simple test script that outputs various code snippets to test the syntax highlighter.
Designed to be used with: python test.py 2>&1 | python highlighter
"""

def print_section(title):
    """Print a section header"""
    print(f"\n==== {title} ====\n")

def main():
    # Bracket tests
    print_section("BRACKET TESTS")
    print("Simple brackets: (test), [test], {test}, <test>")
    print("Nested brackets: (nested (brackets)), [nested [brackets]], {nested {braces}}")
    print("Mixed brackets: {[()]}, ({{[()]}})")
    print("The original example: a = {('a', 'b', 'd[d]'), [('f')]}")
    print("Complex nesting: ((([[[{{{<< nested >>}}}]]]))))")
    
    # String tests
    print_section("STRING TESTS")
    print('String with double quotes: "This is a test"')
    print("String with single quotes: 'This is another test'")
    print('Escaped quotes: "He said \\"Hello\\""')
    print("Escaped quotes: 'Don\\'t worry'")
    print('Mixed quotes: "This has \'single\' quotes"')
    print("Mixed quotes: 'This has \"double\" quotes'")
    print('Brackets in strings: "This has [brackets] in it"')
    print("Incomplete string: \"This string doesn't end")
    
    # Number tests
    print_section("NUMBER TESTS")
    print("Integers: 42, 1000, -273")
    print("Decimals: 3.14159, -0.5, 1.0")
    print("Scientific notation: 6.022e23, 1.6e-19")
    print("Mixed with text: There are 365 days in a year and pi is about 3.14")
    
    # Operator tests
    print_section("OPERATOR TESTS")
    print("Arithmetic: a + b - c * d / e % f")
    print("Assignment: x = y, a += b, c -= d, e *= f, g /= h")
    print("Comparison: a == b, c != d, e < f, g > h, i <= j, k >= l")
    print("Bitwise: a & b | c ^ d ~ e << f >> g")
    print("Logical: a && b || !c")
    
    # File reference tests
    print_section("FILE REFERENCE TESTS")
    print("Basic: [main.py | l.42]")
    print("With path: [src/utils/helper.js | l.105]")
    print("Multiple: [file1.txt | l.10] and [file2.csv | l.20]")
    print("In an error message: Error occurred at [/path/to/file.py | l.789]")
    
    # Python keyword tests
    print_section("PYTHON KEYWORD TESTS")
    print("def function(args):")
    print("if condition: pass")
    print("for item in items:")
    print("while True:")
    print("try: something() except Exception as e:")
    print("class MyClass:")
    print("return result")
    print("from module import thing")
    print("with open('file') as f:")
    print("async def coroutine():")
    print("await future")
    print("lambda x: x * 2")
    print("x is None or x is not None")
    print("global var; nonlocal var")
    print("assert condition, 'message'")
    print("raise Exception('error')")
    
    # Complex examples
    print_section("COMPLEX EXAMPLES")
    print("""
def factorial(n):
    "Calculate the factorial of n"
    if n <= 1:
        return 1
    else:
        return n * factorial(n - 1)

result = factorial(5)  # Should be 120
    """)
    
    print("""
complex_data = {
    "users": [
        {"id": 1, "name": "Alice", "age": 30, "active": True},
        {"id": 2, "name": "Bob", "age": 25, "active": False},
        {"id": 3, "name": "Charlie", "age": 35, "active": True}
    ],
    "statistics": {
        "average_age": 30.0,
        "active_users": 2,
        "inactive_users": 1
    },
    "metadata": {
        "generated_at": "2023-04-15",
        "source_file": "[users.json | l.42]"
    }
}
    """)
    
    print("""
try:
    result = divide(10, 0)
except ZeroDivisionError as e:
    print(f"Error at [calculator.py | l.25]: {e}")
    logging.error("Division by zero detected in function")
    raise ValueError("Invalid input provided")
    """)
# test errors
    print(a)

if __name__ == "__main__":
    main()
