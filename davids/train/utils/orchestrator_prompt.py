"""Few-shot examples for orchestrator prompts."""
import ast
import re
from typing import List, Dict, Tuple, Optional
import random

MATH_FEW_SHOT_EXAMPLES = [
    {
        "user_question": "Find the roots of the quadratic equation: x^2 - 5x + 6 = 0.",
        "subtasks": [
            "Factorize the quadratic equation or apply the quadratic formula to find potential values for x.",
            "Calculate the specific roots, verify them by substituting back into the original equation, and output the final set of solutions."
        ],
        "role_instructions": [
            "You are an algebra expert. Break down the equation into factors or set up the formula correctly.",
            "You are a verifier. Compute the final values and strictly validate them against the equation to ensure correctness."
        ]
    },
    {
        "user_question": "Calculate the hypotenuse of a right-angled triangle with legs of length 5 cm and 12 cm.",
        "subtasks": [
            "Apply the Pythagorean theorem (a^2 + b^2 = c^2) with the given leg lengths.",
            "Compute the square root of the sum to find the hypotenuse and output the final numerical value with units."
        ],
        "role_instructions": [
            "You are a geometry solver. Identify the correct theorem and set up the calculation.",
            "You are a precise calculator. Perform the arithmetic and format the final answer correctly."
        ]
    },
    {
        "user_question": "What is the sum of the first 10 terms of the arithmetic sequence: 2, 5, 8, ...?",
        "subtasks": [
            "Identify the first term (a), common difference (d), and number of terms (n), then select the arithmetic series sum formula.",
            "Substitute the values into the formula S_n = n/2 * (2a + (n-1)d), calculate the result, and output the final integer sum."
        ],
        "role_instructions": [
            "You are a sequence analyst. Extract key parameters from the sequence.",
            "You are a computational solver. Execute the formula and provide the verified final total."
        ]
    },
    {
        "user_question": "If 3x + 2y = 16 and x - y = 2, what are the values of x and y?",
        "subtasks": [
            "Isolate one variable from the second equation (e.g., x = y + 2).",
            "Substitute the isolated expression into the first equation to solve for the first variable.",
            "Substitute the found value back to find the second variable, verify both satisfy the system, and output the solution pair (x, y)."
        ],
        "role_instructions": [
            "You are a strategist. Choose the most efficient method (substitution or elimination) to start.",
            "You are an algebraic solver. Perform the substitution and solve the linear equation.",
            "You are a solution validator. Calculate the second variable and check both against the original system for the final output."
        ]
    },
    {
        "user_question": "A store sells apples for $2 and oranges for $3. Alice bought 10 fruits in total and spent $24. How many apples did she buy?",
        "subtasks": [
            "Define variables for apples (a) and oranges (o) and set up a system of two linear equations based on quantity and cost.",
            "Solve the system of equations to find the values of 'a' and 'o'.",
            "Verify that the sum of fruits is 10 and total cost is $24, then output the final count of apples."
        ],
        "role_instructions": [
            "You are a modeler. Translate the word problem into mathematical equations.",
            "You are a solver. Execute the algebraic steps to find the variables.",
            "You are a logic checker. Ensure the result makes sense in the context of the problem and provide the final answer."
        ]
    },
    {
        "user_question": "Calculate the area of a circle inscribed in a square with a side length of 8 meters.",
        "subtasks": [
            "Determine the diameter and radius of the circle based on the square's side length.",
            "Apply the area formula A = πr^2 using the calculated radius.",
            "Compute the final area value (using π ≈ 3.14159), and output the result rounded to two decimal places."
        ],
        "role_instructions": [
            "You are a geometric analyst. Derive the circle's dimensions from the square.",
            "You are a formula applicator. Set up the area calculation.",
            "You are a precise calculator. Perform the multiplication and format the final output with units."
        ]
    },
    {
        "user_question": "A car travels 60 miles at 30 mph and then another 60 miles at 60 mph. What is the average speed for the entire trip?",
        "subtasks": [
            "Calculate the time taken for the first segment of the trip (Distance / Speed).",
            "Calculate the time taken for the second segment of the trip (Distance / Speed).",
            "Calculate the total distance traveled and the total time taken.",
            "Compute the average speed by dividing total distance by total time, and output the final value."
        ],
        "role_instructions": [
            "You are a physics calculator. Compute time for interval A.",
            "You are a physics calculator. Compute time for interval B.",
            "You are an aggregator. Sum up total metrics.",
            "You are an analyst. Calculate the final harmonic mean (average speed) and verify it's not just the arithmetic mean of speeds."
        ]
    },
    {
        "user_question": "How many distinct permutations can be made from the letters of the word 'MISSISSIPPI'?",
        "subtasks": [
            "Count the total number of letters in the word.",
            "Count the frequency of each repeating letter (M, I, S, P).",
            "Set up the permutation formula n! / (n1! * n2! * ... ) accounting for repetitions.",
            "Calculate the factorials and the final division to output the exact number of distinct permutations."
        ],
        "role_instructions": [
            "You are a data counter. Analyze the input string length.",
            "You are a pattern recognizer. Identify and count duplicates.",
            "You are a combinatorics expert. Construct the correct formula.",
            "You are a number cruncher. Compute the large factorials and final integer result."
        ]
    },
    {
        "user_question": "Find the local maximum value of the function f(x) = -x^2 + 4x + 5.",
        "subtasks": [
            "Differentiate the function f(x) to find f'(x).",
            "Set f'(x) = 0 and solve for x to find critical points.",
            "Verify that the critical point is a maximum using the second derivative test or vertex formula.",
            "Substitute the critical value x back into f(x) to calculate and output the maximum value."
        ],
        "role_instructions": [
            "You are a calculus expert. Perform differentiation.",
            "You are an equation solver. Find the roots of the derivative.",
            "You are an analytical verifier. Confirm the nature of the critical point (min/max).",
            "You are a function evaluator. Compute the final y-value."
        ]
    },
    {
        "user_question": "A bag contains 5 red balls and 3 blue balls. Two balls are drawn without replacement. What is the probability that both are red?",
        "subtasks": [
            "Determine the total number of balls and the probability of drawing a red ball first.",
            "Determine the remaining number of balls and red balls, then calculate the probability of drawing a red ball second.",
            "Multiply the two probabilities to find the joint probability.",
            "Convert the result to a simplified fraction or decimal and output the final probability."
        ],
        "role_instructions": [
            "You are a probability analyst. Calculate the initial state probability.",
            "You are a state tracker. Update counts and calculate conditional probability.",
            "You are a mathematician. Perform the multiplication rule for dependent events.",
            "You are a formatter. Simplify and present the final statistical result."
        ]
    }
]

CODE_FEW_SHOT_EXAMPLES = [
    {
        "user_question": "Write a Python function to check if a given string is a palindrome.",
        "subtasks": [
            "Implement the logic to clean the string (remove spaces/punctuation) and compare it with its reverse.",
            "Combine the logic into a function, add test cases with 'Racecar' and 'hello', and output the complete executable code."
        ],
        "role_instructions": [
            "You are a logic designer. Define the algorithm to normalize and reverse the string.",
            "You are a Python developer. Write the final function and include assert statements to verify correctness."
        ]
    },
    {
        "user_question": "Create a SQL query to find the top 5 highest paid employees from the 'Salary' table.",
        "subtasks": [
            "Identify the correct SQL clauses (SELECT, ORDER BY, LIMIT) to sort data in descending order.",
            "Construct the final SQL query string and verify it against standard SQL syntax constraints."
        ],
        "role_instructions": [
            "You are a database architect. Determine the most efficient sorting method.",
            "You are a SQL specialist. Write the exact query and ensure it is syntactically correct."
        ]
    },
    {
        "user_question": "Write a CSS snippet to perfectly center a div within its parent container.",
        "subtasks": [
            "Define the parent container styles using Flexbox properties (justify-content, align-items).",
            "Write the complete HTML/CSS block showing the parent and child classes, ensuring the child is centered."
        ],
        "role_instructions": [
            "You are a frontend designer. Choose the modern Flexbox approach for alignment.",
            "You are a web developer. Generate the final copy-pasteable code snippet."
        ]
    },
    {
        "user_question": "Create a simple REST API endpoint using Flask that returns 'Hello, World!' in JSON format.",
        "subtasks": [
            "Import the necessary Flask modules and initialize the application instance.",
            "Define a route '/hello' and a view function that returns a JSON dictionary.",
            "Assemble the complete script including the 'if __name__ == \"__main__\":' block and verify it runs on port 5000."
        ],
        "role_instructions": [
            "You are a backend architect. Set up the framework structure.",
            "You are an API developer. Implement the routing and response logic.",
            "You are a deployment engineer. ensure the code is self-contained and executable."
        ]
    },
    {
        "user_question": "Implement a React button component that toggles its text between 'ON' and 'OFF' when clicked.",
        "subtasks": [
            "Import 'useState' from React and define the functional component structure.",
            "Implement the state variable for the toggle status and the 'handleClick' function to invert the state.",
            "Return the JSX code with the onClick event attached, and verify the component exports correctly."
        ],
        "role_instructions": [
            "You are a UI architect. Design the component state management.",
            "You are a React developer. Write the event handling logic.",
            "You are a frontend lead. Assemble the final JSX and export statement."
        ]
    },
    {
        "user_question": "Write a Python script to sort a list of dictionaries by a specific key 'age'.",
        "subtasks": [
            "Create a sample list of dictionaries containing 'name' and 'age' keys.",
            "Use the 'sorted()' function or 'list.sort()' with a lambda function as the key argument.",
            "Output the complete script with the sorted list printed to the console to verify the order."
        ],
        "role_instructions": [
            "You are a data engineer. Prepare the mock data structure.",
            "You are a Python expert. Implement the sorting logic using lambda.",
            "You are a code tester. Combine parts into a runnable script and verify output."
        ]
    },
    {
        "user_question": "Create a Dockerfile to containerize a NodeJS application.",
        "subtasks": [
            "Select the appropriate base image (e.g., node:14-alpine) and set the working directory.",
            "Copy 'package.json' and install dependencies using 'npm install'.",
            "Copy the rest of the application source code and expose the application port.",
            "Define the entry point command (CMD) and output the full Dockerfile content for verification."
        ],
        "role_instructions": [
            "You are a DevOps engineer. Choose the optimal base OS.",
            "You are a build engineer. Handle dependency management layers.",
            "You are a systems integrator. Configure file systems and networking.",
            "You are a release manager. Finalize the configuration file."
        ]
    },
    {
        "user_question": "Implement a Binary Search Tree (BST) insertion method in Java.",
        "subtasks": [
            "Define the 'Node' class with integer data, left child, and right child pointers.",
            "Create the main 'BST' class and implement the recursive 'insert' logic.",
            "Handle edge cases such as inserting into an empty tree or inserting duplicate values.",
            "Write a 'main' method to insert a series of numbers and print the tree (in-order traversal) to verify structure."
        ],
        "role_instructions": [
            "You are a data structure expert. Define the object model.",
            "You are a Java developer. Implement the recursive algorithm.",
            "You are a logic validator. Ensure robustness against edge cases.",
            "You are a QA engineer. Create the test harness and verify the output."
        ]
    },
    {
        "user_question": "Write a Python script using Pandas to read a CSV file, fill missing values with the mean, and save the result.",
        "subtasks": [
            "Import the pandas library and load the CSV file into a DataFrame.",
            "Identify columns with missing values and calculate the mean of those columns.",
            "Fill the NaN values with the calculated mean.",
            "Save the cleaned DataFrame to a new CSV file and output the full code block."
        ],
        "role_instructions": [
            "You are a data analyst. Load and inspect the raw data.",
            "You are a statistician. Determine the imputation strategy.",
            "You are a data engineer. Apply the transformation.",
            "You are a pipeline developer. Finalize the script and handle IO operations."
        ]
    },
    {
        "user_question": "Create a JavaScript function using 'fetch' to get user data from an API and handle errors.",
        "subtasks": [
            "Define an async function that accepts a URL parameter.",
            "Implement the 'fetch' call with 'await' and check the response status (response.ok).",
            "Parse the JSON response if successful, or throw an error if the status is bad.",
            "Wrap the call in a try-catch block, log the results or errors, and output the final functional code snippet."
        ],
        "role_instructions": [
            "You are a JS architect. Setup the asynchronous structure.",
            "You are a network specialist. Implement the HTTP request logic.",
            "You are a defensive coder. Handle JSON parsing and custom errors.",
            "You are a full-stack developer. Assemble the robust function with error handling."
        ]
    }
]



ORCHESTRATOR_PROMPT = """Your role as an assistant involves obtaining answers to questions by an iterative process of querying language models, each with a different role.

You are given a user-provided question. Your objective is to output a sequence of up to {num_agents} workflow steps.

Each step is made of two elements: A subtask and a role instruction for the agent that will handle that subtask.

A subtask could directly ask the language model to solve the given question from scratch, refine the solution of the previous subtask in the sequence, or perform any other completely different task that would facilitate later language models in the sequence to answer the original question with their expertise.

The role instruction should specify what role or expertise the agent should take when solving this subtask.

IMPORTANT: The LAST subtask in your sequence MUST explicitly ask the agent to provide the final answer to the original question. The final agent's output will be used as the solution.

Based on your answer, the first agent will be prompted with the user question, the first subtask, and the first role instruction. Each following agent in the sequence will be prompted with the full history of the previous subtasks and responses, and will be asked to accomplish its relative subtask with its assigned role.

The answer of the final agent will be provided back as the final solution to the user.

Your response should be provided as two Python lists.

The first list should be called subtasks, and contain the strings that will be used to prompt each agent.

The second list should be called role_instructions, and contain the role instruction strings for each corresponding agent.

For instance:

{few_shot_examples}

USER QUESTION:

{instruction}

MAXIMUM NUMBER OF WORKFLOW STEPS: {num_agents}

Please provide your response as two Python lists: subtasks and role_instructions.
""" 


def format_few_shot_example(example: Dict) -> str:
    """
    Format a few-shot example into the prompt format.
    
    Args:
        example: Dictionary with 'user_question', 'subtasks', 'role_instructions'
        
    Returns:
        Formatted string representation of the example
    """
    formatted = f"USER QUESTION:\n{example['user_question']}\n\n"
    
    formatted += "subtasks = [\n"
    for subtask in example['subtasks']:
        # Escape quotes and newlines
        escaped_subtask = subtask.replace('"', '\\"').replace('\n', '\\n')
        formatted += f'    "{escaped_subtask}",\n'
    formatted += "]\n\n"
    
    formatted += "role_instructions = [\n"
    for role_instruction in example['role_instructions']:
        # Escape quotes and newlines
        escaped_role = role_instruction.replace('"', '\\"').replace('\n', '\\n')
        formatted += f'    "{escaped_role}",\n'
    formatted += "]\n"
    
    return formatted


def get_math_few_shot_examples(num_examples: int = 3) -> str:
    """
    Get few-shot examples formatted for the prompt.
    
    Args:
        num_examples: Number of examples to include
        
    Returns:
        Formatted string with few-shot examples
    """
    examples = random.sample(MATH_FEW_SHOT_EXAMPLES, num_examples)
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append(format_few_shot_example(example))
    
    return "\n\n---\n\n".join(formatted_examples)

def get_code_few_shot_examples(num_examples: int = 3) -> str:
    """
    Get few-shot examples formatted for the prompt.
    
    Args:
        num_examples: Number of examples to include
        
    Returns:
        Formatted string with few-shot examples
    """
    examples = random.sample(CODE_FEW_SHOT_EXAMPLES, num_examples)
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append(format_few_shot_example(example))
    
    return "\n\n---\n\n".join(formatted_examples)

def get_orchestrator_prompt(instruction: str, num_agents: int, type: str, num_few_shot_examples: int = 3) -> str:
    """
    Get orchestrator prompt formatted for the prompt.
    
    Args:
        instruction: The user question/problem
        num_agents: Number of agents
        num_few_shot_examples: Number of few-shot examples
        
    Returns:
        Formatted string with orchestrator prompt
    """
    
    if type == "math":
        few_shot_examples = get_math_few_shot_examples(num_few_shot_examples)
    elif type == "code":
        few_shot_examples = get_code_few_shot_examples(num_few_shot_examples)
    else:
        raise ValueError(f"Invalid type: {type}")
    
    return ORCHESTRATOR_PROMPT.format(instruction=instruction, num_agents=num_agents, few_shot_examples=few_shot_examples)


def parse_orchestrator_output(completion: str) -> Optional[Tuple[List[str], List[str]]]:
    """
    Parse orchestrator output to extract subtasks and role_instructions.
    
    Args:
        completion: The orchestrator's completion text
        
    Returns:
        Tuple of (subtasks, role_instructions) if parsing succeeds, None otherwise
    """
    # Try to find subtasks and role_instructions assignments
    subtasks_match = re.search(r'subtasks\s*=\s*\[(.*?)\]', completion, re.DOTALL)
    role_instructions_match = re.search(r'role_instructions\s*=\s*\[(.*?)\]', completion, re.DOTALL)
    
    if subtasks_match and role_instructions_match:
        # Try to parse as Python literals
        subtasks_str = '[' + subtasks_match.group(1) + ']'
        role_instructions_str = '[' + role_instructions_match.group(1) + ']'
        
        try:
            subtasks = ast.literal_eval(subtasks_str)
            role_instructions = ast.literal_eval(role_instructions_str)
            
            if isinstance(subtasks, list) and isinstance(role_instructions, list):
                if len(subtasks) == len(role_instructions) and len(subtasks) > 0:
                    return (subtasks, role_instructions)
        except (ValueError, SyntaxError):
            return None
    
    return None