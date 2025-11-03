"""BigCodeBench evaluation utilities - sanitize and unittest-based evaluation."""
import os
import sys
import multiprocessing
import time
import types
import unittest
from multiprocessing import Value, Manager
from typing import Tuple, Dict, Optional
import numpy as np
import logging

# Import BigCodeBench evaluation utilities
# We'll adapt the code from qwencoder-eval/instruct/BigCodeBench

logger = logging.getLogger(__name__)

# Constants from BigCodeBench eval
TIMEOUT_LIMIT = 240.0
PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def sanitize_code(code: str, entrypoint: Optional[str] = None) -> str:
    """
    Sanitize code by extracting only the necessary functions/classes.
    Uses BigCodeBench's sanitize function if available, otherwise uses simple extraction.
    """
    try:
        # Try to import sanitize from qwencoder-eval
        qwencoder_path = os.path.join(
            os.path.dirname(__file__), 
            '../../qwencoder-eval/instruct/BigCodeBench'
        )
        if os.path.exists(qwencoder_path):
            sys.path.insert(0, qwencoder_path)
            from sanitize import sanitize as bigcodebench_sanitize
            sanitized = bigcodebench_sanitize(code, entrypoint=entrypoint)
            sys.path.pop(0)
            return sanitized
    except (ImportError, Exception) as e:
        logger.debug(f"Could not use BigCodeBench sanitize: {e}")
    
    # Fallback: simple extraction
    import re
    # Extract code blocks if present
    code_block_match = re.search(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
    
    # If entrypoint is provided, try to extract relevant function/class
    if entrypoint:
        # Try to find the function/class definition
        pattern = rf'(?:def|class)\s+{re.escape(entrypoint)}[^:]*:'
        match = re.search(pattern, code, re.MULTILINE)
        if match:
            # Extract from the match to end of function/class
            start = match.start()
            lines = code[start:].split('\n')
            extracted = [lines[0]]
            indent_level = len(lines[0]) - len(lines[0].lstrip())
            
            for line in lines[1:]:
                if line.strip() and len(line) - len(line.lstrip()) <= indent_level:
                    break
                extracted.append(line)
            return '\n'.join(extracted)
    
    return code.strip()


def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Array (dict)
):
    """Execute code in a safe environment with unittest."""
    try:
        # Import BigCodeBench eval utilities
        qwencoder_path = os.path.join(
            os.path.dirname(__file__), 
            '../../qwencoder-eval/instruct/BigCodeBench'
        )
        if os.path.exists(qwencoder_path):
            sys.path.insert(0, qwencoder_path)
            from eval.utils import (
                create_tempdir,
                reliability_guard,
                swallow_io,
                time_limit,
                safe_environment,
            )
            use_bigcodebench_utils = True
        else:
            use_bigcodebench_utils = False
            logger.warning("BigCodeBench eval utils not found, using basic execution")
        
        if use_bigcodebench_utils:
            import os as os_module
            import shutil
            import builtins
            
            with safe_environment(), create_tempdir():
                rmtree = shutil.rmtree
                rmdir = os_module.rmdir
                chdir = os_module.chdir
                
                # Disable functionalities that can make destructive changes
                reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
                module_name = "__test__"
                new_module = types.ModuleType(module_name)
                
                # Set necessary attributes for the module
                new_module.__dict__.update({
                    '__builtins__': builtins,
                    '__file__': f"{module_name}.py",
                    '__package__': None,
                    '__doc__': None,
                    'sys': sys,
                    'os': os_module,
                    'environ': os_module.environ,
                })

                try:
                    full_code = code + "\n" + test_code

                    with swallow_io():
                        exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                        sys.modules[module_name] = new_module
                        TestCases = getattr(new_module, 'TestCases')
                        loader = unittest.TestLoader()
                        suite = loader.loadTestsFromTestCase(TestCases)
                        test_result = unittest.TestResult()
                        with time_limit(timeout):
                            suite.run(test_result)

                    issues = test_result.failures + test_result.errors
                    for test, trace in issues:
                        details[test.id().split(".")[-1]] = str(trace)
                    stat.value = _SUCCESS
                except BaseException as e:
                    details["ALL"] = str(e)
                    stat.value = _FAILED
                finally:
                    # Needed for cleaning up
                    shutil.rmtree = rmtree
                    os_module.rmdir = rmdir
                    os_module.chdir = chdir
        else:
            # Fallback: basic execution without full sandboxing
            # WARNING: This is less safe but allows evaluation even without BigCodeBench utils
            import tempfile
            import shutil
            with tempfile.TemporaryDirectory() as tmpdir:
                import os
                os.chdir(tmpdir)
                try:
                    full_code = code + "\n" + test_code
                    exec(compile(full_code, "__test__.py", 'exec'), {
                        '__builtins__': __builtins__,
                        '__name__': '__test__',
                        '__file__': '__test__.py',
                    })
                    # Try to run tests
                    import unittest
                    test_module = sys.modules.get('__test__')
                    if test_module and hasattr(test_module, 'TestCases'):
                        loader = unittest.TestLoader()
                        suite = loader.loadTestsFromTestCase(test_module.TestCases)
                        test_result = unittest.TestResult()
                        suite.run(test_result)
                        if test_result.wasSuccessful():
                            stat.value = _SUCCESS
                        else:
                            for test, trace in test_result.failures + test_result.errors:
                                details[test.id().split(".")[-1]] = str(trace)
                            stat.value = _FAILED
                    else:
                        stat.value = _FAILED
                        details["ALL"] = "TestCases class not found"
                except BaseException as e:
                    details["ALL"] = str(e)
                    stat.value = _FAILED
    except Exception as e:
        logger.error(f"Error in unsafe_execute: {e}")
        details["ALL"] = str(e)
        stat.value = _FAILED
    finally:
        if 'qwencoder_path' in locals() and qwencoder_path in sys.path:
            sys.path.remove(qwencoder_path)


def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float = 128 * 1024,
    max_data_limit: float = 4 * 1024,
    max_stack_limit: float = 5,
    min_time_limit: float = 1.0,
    gt_time_limit: float = 20.0,
) -> Tuple[str, Dict]:
    """
    Check if code passes all unittest tests.
    
    Returns:
        (status, details) where status is "pass", "fail", or "timeout"
        details is a dict mapping test names to error messages
    """
    time_limit_val = max(min_time_limit, gt_time_limit)
    timeout = max(float(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT)), time_limit_val) + 1
    
    # Shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat_val = _mapping[stat.value]
    # Convert details to a dict
    details_dict = dict(details)

    if not stat_val:
        stat_val = TIMEOUT
    if stat_val == PASS:
        if details_dict:
            stat_val = FAIL

    return stat_val, details_dict

