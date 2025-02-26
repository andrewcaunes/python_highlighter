#!/usr/bin/env python3
import re
import sys

# ANSI color codes for terminal output
class Color:
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    GRAY = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuration for syntax highlighting
BRACKET_COLORS = [
    Color.RED,
    Color.GREEN,
    Color.BLUE, 
    Color.MAGENTA,
    Color.CYAN,
    Color.YELLOW,
    Color.BRIGHT_RED,
    Color.BRIGHT_GREEN,
    Color.BRIGHT_BLUE,
    Color.BRIGHT_MAGENTA,
    Color.BRIGHT_CYAN,
    Color.BRIGHT_YELLOW
]

# Default priorities for syntax elements (higher numbers = higher priority)
SYNTAX_PRIORITIES = {
    'STRING': 100,
    'BRACKET': 50,
    'NUMBER': 30,
    'OPERATOR': 20,
    'FILE_REF': 80,
    'PATH': 40,
    'KEYWORD': 10,
    'FUNCTION': 15,
    'CLASS': 15,
    'MODULE': 15,
    'TRACEBACK': 200,
    'ERROR_NAME': 220,
    'ERROR_MSG': 215,
    'PATH_IN_FILE_REF': 90,
}

STRING_COLOR = Color.BRIGHT_GREEN
NUMBER_COLOR = Color.BRIGHT_YELLOW
OPERATOR_COLOR = Color.BRIGHT_MAGENTA
FILE_REF_COLOR = Color.BLUE
PATH_COLOR = Color.BLUE
KEYWORD_COLOR = Color.BRIGHT_CYAN
FUNCTION_COLOR = Color.BRIGHT_BLUE
CLASS_COLOR = Color.BRIGHT_YELLOW
MODULE_COLOR = Color.CYAN
TRACEBACK_COLOR = Color.BRIGHT_RED
ERROR_NAME_COLOR = f"{Color.BRIGHT_RED}{Color.BOLD}"
ERROR_MSG_COLOR = Color.BRIGHT_WHITE

# Lists of built-in types and common items for accurate highlighting
BUILTIN_EXCEPTIONS = [
    'BaseException', 'Exception', 'ArithmeticError', 'BufferError', 'LookupError',
    'AssertionError', 'AttributeError', 'EOFError', 'FloatingPointError',
    'GeneratorExit', 'ImportError', 'ModuleNotFoundError', 'IndexError', 'KeyError',
    'KeyboardInterrupt', 'MemoryError', 'NameError', 'NotImplementedError',
    'OSError', 'OverflowError', 'RecursionError', 'ReferenceError', 'RuntimeError',
    'StopIteration', 'StopAsyncIteration', 'SyntaxError', 'IndentationError',
    'TabError', 'SystemError', 'SystemExit', 'TypeError', 'UnboundLocalError',
    'UnicodeError', 'UnicodeEncodeError', 'UnicodeDecodeError', 'UnicodeTranslateError',
    'ValueError', 'ZeroDivisionError', 'EnvironmentError', 'IOError',
    'WindowsError', 'BlockingIOError', 'ChildProcessError', 'ConnectionError',
    'BrokenPipeError', 'ConnectionAbortedError', 'ConnectionRefusedError',
    'ConnectionResetError', 'FileExistsError', 'FileNotFoundError',
    'InterruptedError', 'IsADirectoryError', 'NotADirectoryError',
    'PermissionError', 'ProcessLookupError', 'TimeoutError'
]

BUILTIN_TYPES = [
    'bool', 'bytearray', 'bytes', 'complex', 'dict', 'float', 'frozenset',
    'int', 'list', 'memoryview', 'object', 'range', 'set', 'slice', 'str',
    'tuple', 'type'
]

COMMON_MODULES = [
    'os', 'sys', 're', 'math', 'random', 'time', 'datetime', 'collections',
    'json', 'csv', 'pickle', 'sqlite3', 'logging', 'argparse', 'pathlib',
    'threading', 'multiprocessing', 'subprocess', 'socket', 'urllib', 'http',
    'requests', 'flask', 'django', 'numpy', 'pandas', 'matplotlib', 'scipy',
    'sklearn', 'tensorflow', 'torch', 'PIL', 'itertools', 'functools',  'absl-py', 
    'addict', 'aliyun-python-sdk-core', 'aliyun-python-sdk-kms', 'annotated-types', 
    'antlr4-python3-runtime', 'argcomplete', 'asttokens', 'attrs', 'black', 'blinker', 
    'Brotli', 'cachetools', 'calmsize', 'certifi', 'cffi', 'chardet', 'charset-normalizer', 
    'click', 'click-plugins', 'cligj', 'cmake', 'colorama', 'comm', 'ConfigArgParse', 'contourpy',
    'crcmod', 'cryptography', 'cycler', 'Cython', 'dash', 'dash-core-components',
    'dash-html-components', 'dash-table', 'debugpy', 'decorator', 'descartes', 'evo',
    'exceptiongroup', 'executing', 'faiss', 'fastjsonschema', 'filelock', 'fiona',
    'fire', 'flake8', 'Flask', 'fonttools', 'freetype-py', 'ftfy', 'future',
    'geopandas', 'gmpy2', 'grpcio', 'hdbscan', 'idna', 'imageio', 'importlib_metadata',
    'iniconfig', 'ipykernel', 'ipython', 'ipywidgets', 'itsdangerous', 'jedi', 'Jinja2',
    'jmespath', 'joblib', 'jsonschema', 'jsonschema-', 'jupyter_client', 'jupyter_core',
    'jupyterlab_widgets', 'kiss-icp', 'kiwisolver', 'laspy', 'lazy_loader', 'llvmlite',
    'lyft-dataset-sdk', 'lz4', 'Markdown', 'markdown-it-py', 'MarkupSafe', 'matplotlib',
    'matplotlib-inline', 'mccabe', 'mdurl', 'mmcv', 'mmdet', 'mmdet3d', 'mmengine',
    'mmsegmentation', 'model-index', 'mpmath', 'multipledispatch', 'mypy-extensions',
    'natsort', 'nbformat', 'nest_asyncio', 'networkx', 'ninja', 'nksr', 'numba',
    'numexpr', 'numpy', 'nuscenes-devkit', 'oauthlib', 'omegaconf', 'open3d',
    'opencv-python', 'opendatalab', 'openmim', 'openxlab', 'ordered-set', 'oss2',
    'packaging', 'pandas', 'parso', 'Paste', 'pathspec', 'pexpect', 'pickleshare',
    'pillow', 'pip', 'platformdirs', 'plotly', 'pluggy', 'plyfile', 'prettytable',
    'prompt-toolkit', 'protobuf', 'psutil', 'ptyprocess', 'pure-eval', 'pybind11',
    'pycocotools', 'pycodestyle', 'pycparser', 'pycryptodome', 'pydantic',
    'pydantic_core', 'pydantic-settings', 'pyflakes', 'pyglet', 'Pygments', 'pykalman',
    'pykdtree', 'pyntcloud', 'pynvml', 'PyOpenGL', 'PyOpenGL-accelerate', 'pyparsing',
    'pypng', 'pyproj', 'pyproject-metadata', 'pyquaternion', 'pyrender', 'pyrr',
    'PySocks', 'pytest', 'python-dateutil', 'python-dotenv', 'python-pycg', 'pytz',
    'PyYAML', 'pyzmq', 'referencing', 'regex', 'reportlab', 'requests', 'retrying',
    'rich', 'rosbags', 'rpds-py', 'ruamel.yaml', 'ruamel.yaml.clib', 'scikit_build_core',
    'scikit-image', 'scikit-learn', 'scipy', 'screeninfo', 'seaborn', 'setuptools',
    'Shapely', 'shellingham', 'six', 'stack-data', 'sympy', 'tabulate', 'tenacity',
    'tensorboard', 'tensorboard-data-server', 'termcolor', 'terminaltables',
    'threadpoolctl', 'tifffile', 'tomli', 'torch', 'torch-scatter', 'torchvision',
    'tornado', 'tqdm', 'traitlets', 'trimesh', 'triton', 'typer', 'typing_extensions',
    'tzdata', 'urllib3', 'usd-core', 'wcwidth', 'Werkzeug', 'wheel',
    'widgetsnbextension', 'yapf', 'zipp', 'zstandard',
]

BUILTIN_FUNCTIONS = [
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
    'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex', 'delattr',
    'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec', 'filter', 'float',
    'format', 'frozenset', 'getattr', 'globals', 'hasattr', 'hash', 'help',
    'hex', 'id', 'input', 'int', 'isinstance', 'issubclass', 'iter', 'len',
    'list', 'locals', 'map', 'max', 'memoryview', 'min', 'next', 'object',
    'oct', 'open', 'ord', 'pow', 'print', 'property', 'range', 'repr', 'reversed',
    'round', 'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum',
    'super', 'tuple', 'type', 'vars', 'zip', '__import__'
]

# Class to represent a syntax element to be highlighted
class SyntaxElement:
    def __init__(self, start, end, color, priority=0):
        self.start = start
        self.end = end
        self.color = color
        self.priority = priority  # Higher priority elements override lower ones
    
    def __lt__(self, other):
        # Sort by priority (higher first), then by start position (earlier first)
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.start < other.start

def detect_strings(text, priority=SYNTAX_PRIORITIES['STRING']):
    """Identify string regions in the text
    
    Returns:
        list: A list of SyntaxElement objects for strings
    """
    elements = []
    i = 0
    while i < len(text):
        if text[i] in "'\"":
            quote_char = text[i]
            start = i
            i += 1
            escaped = False
            while i < len(text):
                if escaped:
                    escaped = False
                elif text[i] == '\\':
                    escaped = True
                elif text[i] == quote_char:
                    elements.append(SyntaxElement(start, i, STRING_COLOR, priority=priority))
                    i += 1
                    break
                i += 1
            if i >= len(text):  # Unclosed string
                elements.append(SyntaxElement(start, len(text) - 1, STRING_COLOR, priority=priority))
        else:
            i += 1
    
    return elements

def detect_brackets(text, string_elements, priority=SYNTAX_PRIORITIES['BRACKET']):
    """Identify bracket pairs in the text
    
    Returns:
        list: A list of SyntaxElement objects for brackets
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    bracket_pairs = {'(': ')', '[': ']', '{': '}', '<': '>'}
    bracket_stacks = {b_type: [] for b_type in bracket_pairs}
    
    for i, char in enumerate(text):
        if in_string[i]:
            continue
        
        if char in bracket_pairs:
            # Opening bracket
            bracket_type = char
            color_idx = len(bracket_stacks[bracket_type]) % len(BRACKET_COLORS)
            bracket_stacks[bracket_type].append((i, color_idx))
        elif char in bracket_pairs.values():
            # Closing bracket
            for open_bracket, close_bracket in bracket_pairs.items():
                if char == close_bracket and bracket_stacks[open_bracket]:
                    open_pos, color_idx = bracket_stacks[open_bracket].pop()
                    color = BRACKET_COLORS[color_idx]
                    elements.append(SyntaxElement(open_pos, open_pos, color, priority=priority))
                    elements.append(SyntaxElement(i, i, color, priority=priority))
                    break
    
    return elements

def detect_numbers(text, string_elements, priority=SYNTAX_PRIORITIES['NUMBER']):
    """Identify numeric literals in the text
    
    Returns:
        list: A list of SyntaxElement objects for numbers
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    number_pattern = r'\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b'
    
    for match in re.finditer(number_pattern, text):
        start, end = match.span()
        # Check if any part of the number is in a string
        if not any(in_string[i] for i in range(start, end)):
            elements.append(SyntaxElement(start, end - 1, NUMBER_COLOR, priority=priority))
    
    return elements

def detect_operators(text, string_elements, priority=SYNTAX_PRIORITIES['OPERATOR']):
    """Identify operators in the text
    
    Returns:
        list: A list of SyntaxElement objects for operators
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    operators = ['+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', 
                '+=', '-=', '*=', '/=', '%=', '&', '|', '^', '~', '<<', '>>', 
                '&&', '||', '!', '++', '--']
    
    # Sort by length to prioritize multi-char operators
    operators = sorted(operators, key=len, reverse=True)
    
    for op in operators:
        escaped_op = re.escape(op)
        # Look for operators with non-alphanumeric characters around them
        pattern = f'(?<![a-zA-Z0-9_]){escaped_op}(?![a-zA-Z0-9_])'
        
        for match in re.finditer(pattern, text):
            start, end = match.span()
            # Check if any part of the operator is in a string
            if not any(in_string[i] for i in range(start, end)):
                elements.append(SyntaxElement(start, end - 1, OPERATOR_COLOR, priority=priority))
    
    return elements

def detect_file_references(text, string_elements, priority=SYNTAX_PRIORITIES['FILE_REF']):
    """Identify file references in the text
    
    Returns:
        list: A list of SyntaxElement objects for file references and contained paths
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    # Pattern: [filename | l.linenumber]
    file_ref_pattern = r'\[([\w\./\-]+(?:/[\w\./\-]+)*)\s*\|\s*l\.(\d+)\]'
    
    for match in re.finditer(file_ref_pattern, text):
        full_start, full_end = match.span()
        
        # Skip if in string
        if any(in_string[i] for i in range(full_start, full_end)):
            continue
            
        # Add the full file reference
        elements.append(SyntaxElement(full_start, full_end - 1, FILE_REF_COLOR, priority=priority))
        
        # Add the path inside (group 1)
        path_text = match.group(1)
        path_start = full_start + 1  # Skip the opening bracket
        path_end = path_start + len(path_text) - 1
        elements.append(SyntaxElement(path_start, path_end, PATH_COLOR, priority=SYNTAX_PRIORITIES['PATH_IN_FILE_REF']))
    
    return elements

def detect_paths(text, string_elements, priority=SYNTAX_PRIORITIES['PATH']):
    """Identify standalone paths in the text
    
    Returns:
        list: A list of SyntaxElement objects for paths
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    
    # Match standalone absolute paths
    abs_path_pattern = r'(?<![a-zA-Z0-9_"\[\]])/(?:[\w\.\-]+/)*[\w\.\-]+'
    
    for match in re.finditer(abs_path_pattern, text):
        start, end = match.span()
        if not any(in_string[i] for i in range(start, end)):
            elements.append(SyntaxElement(start, end - 1, PATH_COLOR, priority=priority))
    
    # Match relative paths starting with ./ or ../
    rel_path_pattern = r'(?<![a-zA-Z0-9_"\[\]])\.\.?/(?:[\w\.\-]+/)*[\w\.\-]+'
    
    for match in re.finditer(rel_path_pattern, text):
        start, end = match.span()
        if not any(in_string[i] for i in range(start, end)):
            elements.append(SyntaxElement(start, end - 1, PATH_COLOR, priority=priority))
    
    # Match multi-directory paths without leading / or .
    multi_dir_pattern = r'(?<![a-zA-Z0-9_"\/\.\[\]])[\w\.\-]+/[\w\.\-]+(?:/[\w\.\-]+)+'
    
    for match in re.finditer(multi_dir_pattern, text):
        start, end = match.span()
        if not any(in_string[i] for i in range(start, end)):
            elements.append(SyntaxElement(start, end - 1, PATH_COLOR, priority=priority))
    
    return elements

def detect_python_keywords(text, string_elements, priority=SYNTAX_PRIORITIES['KEYWORD']):
    """Identify Python keywords in the text
    
    Returns:
        list: A list of SyntaxElement objects for keywords
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    keywords = [
        'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
        'def', 'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
        'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
        'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'True', 'try',
        'while', 'with', 'yield'
    ]
    
    for keyword in keywords:
        pattern = fr'\b{keyword}\b'
        for match in re.finditer(pattern, text):
            start, end = match.span()
            # Check if any part of the keyword is in a string
            if not any(in_string[i] for i in range(start, end)):
                elements.append(SyntaxElement(start, end - 1, KEYWORD_COLOR, priority=priority))
    
    return elements

def detect_function_calls(text, string_elements, priority=SYNTAX_PRIORITIES['FUNCTION']):
    """Identify function calls and definitions in the text
    
    Returns:
        list: A list of SyntaxElement objects for function calls and definitions
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    
    # Function definitions (def name)
    def_pattern = r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(def_pattern, text):
        func_start, func_end = match.span(1)  # Group 1 is the function name
        if not any(in_string[i] for i in range(func_start, func_end)):
            elements.append(SyntaxElement(func_start, func_end - 1, FUNCTION_COLOR, priority=priority))
    
    # Function/method calls (name(...))
    call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    for match in re.finditer(call_pattern, text):
        func_name = match.group(1)
        func_start, func_end = match.span(1)
        
        # Skip if in string or if the name is a Python keyword
        if any(in_string[i] for i in range(func_start, func_end)) or func_name in ['if', 'for', 'while', 'with']:
            continue
        
        # Use a different color for built-in functions
        if func_name in BUILTIN_FUNCTIONS:
            elements.append(SyntaxElement(func_start, func_end - 1, FUNCTION_COLOR, priority=priority))
        else:
            elements.append(SyntaxElement(func_start, func_end - 1, FUNCTION_COLOR, priority=priority))
    
    return elements

def detect_classes(text, string_elements, priority=SYNTAX_PRIORITIES['CLASS']):
    """Identify class names, definitions, and instantiations
    
    Returns:
        list: A list of SyntaxElement objects for class names
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    
    # Class definitions
    class_def_pattern = r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(class_def_pattern, text):
        class_start, class_end = match.span(1)  # Group 1 is the class name
        if not any(in_string[i] for i in range(class_start, class_end)):
            elements.append(SyntaxElement(class_start, class_end - 1, CLASS_COLOR, priority=priority))
    
    # Exception classes and built-in types
    for cls in BUILTIN_EXCEPTIONS + BUILTIN_TYPES:
        pattern = fr'\b{cls}\b'
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if not any(in_string[i] for i in range(start, end)):
                elements.append(SyntaxElement(start, end - 1, CLASS_COLOR, priority=priority))
    
    return elements

def detect_modules(text, string_elements, priority=SYNTAX_PRIORITIES['MODULE']):
    """Identify module/library names
    
    Returns:
        list: A list of SyntaxElement objects for module names
    """
    # Create a lookup for string regions
    in_string = [False] * len(text)
    for elem in string_elements:
        for i in range(elem.start, elem.end + 1):
            in_string[i] = True
    
    elements = []
    
    # Import statements
    import_patterns = [
        r'\bimport\s+([a-zA-Z_][a-zA-Z0-9_]*)',  # import module
        r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import'  # from module import
    ]
    
    for pattern in import_patterns:
        for match in re.finditer(pattern, text):
            mod_start, mod_end = match.span(1)  # Group 1 is the module name
            if not any(in_string[i] for i in range(mod_start, mod_end)):
                elements.append(SyntaxElement(mod_start, mod_end - 1, MODULE_COLOR, priority=priority))
    
    # Module usage (module.something)
    for module in COMMON_MODULES:
        # Look for module name followed by dot
        pattern = fr'\b{module}\.'
        for match in re.finditer(pattern, text):
            start, end = match.span()
            # Exclude the dot at the end
            end = end - 1
            if not any(in_string[i] for i in range(start, end)):
                elements.append(SyntaxElement(start, end - 1, MODULE_COLOR, priority=priority))
    
    return elements

def detect_errors(text, priority_traceback=SYNTAX_PRIORITIES['TRACEBACK'], 
                  priority_error_name=SYNTAX_PRIORITIES['ERROR_NAME'], 
                  priority_error_msg=SYNTAX_PRIORITIES['ERROR_MSG']):
    """Identify Python error messages and tracebacks
    
    Returns:
        list: A list of SyntaxElement objects for error information
    """
    elements = []
    
    # Traceback header
    traceback_pattern = r'\bTraceback \(most recent call last\):'
    for match in re.finditer(traceback_pattern, text):
        start, end = match.span()
        elements.append(SyntaxElement(start, end - 1, TRACEBACK_COLOR, priority=priority_traceback))
    
    # File references in tracebacks
    file_ref_pattern = r'File "([^"]+)", line (\d+)'
    for match in re.finditer(file_ref_pattern, text):
        full_start, full_end = match.span()
        elements.append(SyntaxElement(full_start, full_end - 1, TRACEBACK_COLOR, priority=priority_traceback))
        
        # Path inside quotes
        path_start, path_end = match.span(1)
        elements.append(SyntaxElement(path_start, path_end - 1, PATH_COLOR, priority=SYNTAX_PRIORITIES['PATH_IN_FILE_REF']))
    
    # Error type and message - looking for ExceptionName: message
    error_pattern = r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception|Warning|Exit))\s*:(.*?)(?:\n|$)'
    for match in re.finditer(error_pattern, text):
        # Error name
        name_start, name_end = match.span(1)
        elements.append(SyntaxElement(name_start, name_end - 1, ERROR_NAME_COLOR, priority=priority_error_name))
        
        # Error message
        msg_start, msg_end = match.span(2)
        if msg_start < msg_end:  # Only if there's an actual message
            elements.append(SyntaxElement(msg_start, msg_end - 1, ERROR_MSG_COLOR, priority=priority_error_msg))
    
    return elements

def apply_highlighting(text, elements):
    """Apply all highlighting to the text based on identified elements
    
    This builds a new string with all the color codes inserted at the right positions.
    """
    # Sort elements by priority (highest first), then by start position
    elements.sort()
    
    # Create a list to track what color should be active at each position
    active_colors = [None] * len(text)
    
    # Apply elements (higher priority overwrites lower)
    for elem in elements:
        for i in range(elem.start, elem.end + 1):
            active_colors[i] = elem.color
    
    # Build the result
    result = []
    current_color = None
    
    for i, char in enumerate(text):
        color = active_colors[i]
        
        # If color changed, add the color code
        if color != current_color:
            # If we had a color before, add reset
            if current_color is not None:
                result.append(Color.RESET)
            
            # If we have a new color, add it
            if color is not None:
                result.append(color)
            
            current_color = color
        
        # Add the character
        result.append(char)
    
    # Reset color at the end if needed
    if current_color is not None:
        result.append(Color.RESET)
    
    return ''.join(result)

def highlight_line(line):
    """Apply syntax highlighting to a line of text"""
    # First detect strings (we need these for all other detectors)
    string_elements = detect_strings(line)
    
    # Detect all other syntax elements
    file_ref_elements = detect_file_references(line, string_elements)
    path_elements = detect_paths(line, string_elements)
    function_elements = detect_function_calls(line, string_elements)
    class_elements = detect_classes(line, string_elements)
    module_elements = detect_modules(line, string_elements)
    bracket_elements = detect_brackets(line, string_elements)
    number_elements = detect_numbers(line, string_elements)
    operator_elements = detect_operators(line, string_elements)
    keyword_elements = detect_python_keywords(line, string_elements)
    error_elements = detect_errors(line)  # No need to pass string_elements, errors aren't inside strings
    
    # Combine all elements
    all_elements = (string_elements + file_ref_elements + path_elements + 
                   function_elements + class_elements + module_elements +
                   bracket_elements + number_elements + operator_elements + 
                   keyword_elements + error_elements)
    
    # Apply highlighting
    return apply_highlighting(line, all_elements)

def main():
    """Process input stream and apply syntax highlighting"""
    if len(sys.argv) > 1 and (sys.argv[1] == '-h' or sys.argv[1] == '--help'):
        print("Usage: python highlight.py < input.txt")
        print("       or pipe output: python your_script.py | python highlight.py")
        sys.exit(0)

    try:
        for line in sys.stdin:
            print(highlight_line(line), end='')
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        sys.exit(0)
    except BrokenPipeError:
        # Handle broken pipe (e.g., piping to 'head')
        sys.stderr.close()
        sys.exit(0)

if __name__ == "__main__":
    main()