# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import multiprocessing
import unittest
import contextlib
import io
import signal
import tempfile
import os
import shutil
import platform
import subprocess
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Helper functions for unittest execution
test_code_unittest = '''
import unittest
runner = unittest.TextTestRunner()
print('Runner has been built...\\n')
run_result = runner.run(test_suite)
print('Runner result acquired...\\n')
total_tests = run_result.testsRun
failed_tests = len(run_result.failures)
error_tests = len(run_result.errors)
passed_tests = total_tests - failed_tests - error_tests
passed_tests = max(0, passed_tests)
pass_rate = passed_tests / total_tests if total_tests > 0 else 0
'''

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore


def safe_bcb_execute(queue: multiprocessing.Queue, program: str, testcase: str, timeout: float):
    """Safely execute the program with unittest testcase."""
    logger.info(f"Starting test execution with timeout={timeout}s")
    
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        rmtree = shutil.rmtree
        rmdir_ = os.rmdir
        chdir_ = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        try:
            # Try to import max_memory_as_bytes, fallback to None if not available
            try:
                max_memory_as_bytes = 16 * 1024 * 1024 * 1024
                reliability_guard(maximum_memory_bytes=max_memory_as_bytes)
            except ImportError:
                reliability_guard(maximum_memory_bytes=None)
        except Exception:
            pass  # Some systems may not support resource limits

        # Construct the check program and run it.
        check_program = program

        try:
            logger.debug("Loading testcase into exec_globals")
            exec_globals = {}
            exec(testcase, exec_globals)
            test_suite = unittest.TestLoader().loadTestsFromTestCase(exec_globals['TestCases'])
            num_tests = test_suite.countTestCases()
            logger.info(f"Loaded test suite with {num_tests} test case(s)")
            assert num_tests > 0, "The testcase must have at least one test method\n"
            exec_globals['test_suite'] = test_suite

            logger.debug("Executing test program")
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            
            pass_rate = exec_globals['pass_rate']
            total_tests = exec_globals.get('total_tests', 0)
            passed_tests = exec_globals.get('passed_tests', 0)
            failed_tests = exec_globals.get('failed_tests', 0)
            error_tests = exec_globals.get('error_tests', 0)
            
            logger.info(
                f"Test execution completed: pass_rate={pass_rate:.4f}, "
                f"total={total_tests}, passed={passed_tests}, failed={failed_tests}, errors={error_tests}"
            )
            result_item = {'score': pass_rate, 'status': 'succeed'}

        except TimeoutException:
            logger.warning(f"Test execution timed out after {timeout}s")
            result_item = {'score': 0, 'status': 'timeout'}
        except Exception as e:
            logger.error(f"Test execution failed with error: {str(e)}", exc_info=True)
            result_item = {'score': 0, 'status': str(e)}

        queue.put(result_item)
        logger.debug("Test result sent to queue")

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir_
        os.chdir = chdir_


def get_bcb_score(program: str, testcase: str, timeout: float, completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. Used for BigCodeBench

    :param testcase: unittest testcase definition
    :param timeout: timeout for execution
    :param program: program code
    :param completion_id: an optional completion ID used for matching
        the results later even if execution finishes asynchronously
    """
    logger.info(f"Starting get_bcb_score (completion_id={completion_id}, timeout={timeout}s)")
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=safe_bcb_execute, args=(q, program, testcase, timeout))

    try:
        logger.debug("Starting test execution process")
        p.start()
        p.join(timeout=timeout + 1)
        if p.is_alive():
            logger.warning("Test process is still alive after timeout, killing it")
            p.kill()
            p.join()
        logger.debug("Retrieving result from queue")
        result_dict = q.get(timeout=timeout + 1)
        logger.info(f"Test result retrieved: score={result_dict.get('score', 0)}, status={result_dict.get('status', 'unknown')}")

    except Exception as e:
        logger.error(f"Error in get_bcb_score: {str(e)}", exc_info=True)
        if p.is_alive():
            p.kill()
            p.join()
        result_dict = {'score': 0, 'status': str(e)}

    status = result_dict['status']
    result = {'score': result_dict['score'], 'status': status, 'completion_id': completion_id}
    return result


def bcb_accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str] = None, **kwargs) -> list[float]:
    r"""
    Reward function that extracts the answer from <answer> tags and verifies it using unittest.
    Returns 1.0 if all tests pass, 0.0 otherwise.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution (`list[str]`, optional):
            List of unittest testcase definitions (as strings) to be executed.
            If not provided, will be extracted from kwargs.
        **kwargs:
            Additional keyword arguments. Can include 'solution' key if not provided as a parameter.
            This is required for compatibility with trainers like [`GRPOTrainer`].
    
    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if all tests pass, 0.0 otherwise.
    
    Example:
    ```python
    >>> testcase = '''
    ... import unittest
    ... class TestCases(unittest.TestCase):
    ...     def test_addition(self):
    ...         self.assertEqual(add(1, 2), 3)
    ... '''
    >>> solution = [testcase]
    >>> completion = [
    ...     [{"role": "assistant", "content": "<answer>\ndef add(a, b):\n    return a + b\n</answer>"}],
    ... ]
    >>> bcb_accuracy_reward(completion, solution)
    [1.0]
    ```
    """
    # Extract solution from kwargs if not provided as parameter
    if solution is None:
        return [0.0] * len(completions)
    
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    logger.info(f"Processing {len(completions)} completion(s) with {len(solution)} testcase(s)")
    
    for idx, (content, testcase) in enumerate(zip(contents, solution, strict=True)):
        logger.info(f"Processing completion {idx + 1}/{len(completions)}")
        
        # Extract answer from <answer>...</answer> tags
        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.search(answer_pattern, content, re.DOTALL)
        
        if not match:
            # If no answer tag found, return 0.0
            logger.warning(f"Completion {idx + 1}: No <answer> tag found")
            rewards.append(0.0)
            continue
        
        extracted_answer = match.group(1).strip()
        logger.debug(f"Completion {idx + 1}: Extracted answer length: {len(extracted_answer)} chars")
        
        # Extract code from markdown code blocks if present
        code_block_pattern = r"```(?:python)?\n?(.*?)```"
        code_match = re.search(code_block_pattern, extracted_answer, re.DOTALL)
        if code_match:
            extracted_answer = code_match.group(1).strip()
            logger.debug(f"Completion {idx + 1}: Extracted code from markdown block, length: {len(extracted_answer)} chars")
        
        # Construct the program with test code
        test_program = extracted_answer + '\n' + test_code_unittest
        logger.debug(f"Completion {idx + 1}: Test program length: {len(test_program)} chars")
        
        # Run unittest and get score
        try:
            result = get_bcb_score(test_program, testcase, timeout=60.0, completion_id=idx)
            # Convert pass_rate to binary: 1.0 if all tests pass (pass_rate == 1.0), 0.0 otherwise
            # reward = 1.0 if result['score'] == 1.0 else 0.0 # binary reward
            reward = result['score'] # continuous reward
            logger.info(
                f"Completion {idx + 1}: Test result - score={result['score']:.4f}, "
                f"status={result['status']}, reward={reward}"
            )
        except Exception as e:
            logger.error(f"Completion {idx + 1}: Exception during test execution: {str(e)}", exc_info=True)
            reward = 0.0
        
        rewards.append(reward)
    
    logger.info(f"All completions processed. Reward summary: {sum(rewards)}/{len(rewards)} passed")
    return rewards