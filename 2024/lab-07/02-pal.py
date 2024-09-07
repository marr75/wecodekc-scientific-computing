import random


# Function to generate 30 difficult floating-point math problems and their answers
def generate_floating_point_problems_with_answers():
    problems_and_answers = []
    for _ in range(30):
        a = round(random.uniform(1000, 10000), 6)
        b = round(random.uniform(1000, 10000), 6)
        operator = random.choice(["+", "-", "*", "/"])

        # Compute the answer based on the operator
        if operator == "+":
            answer = a + b
        elif operator == "-":
            answer = a - b
        elif operator == "*":
            answer = a * b
        else:  # operator == "/"
            answer = a / b if b != 0 else "undefined"  # Avoid division by zero

        # Create the problem and answer pair
        problem = f"{a} {operator} {b}"
        problems_and_answers.append((problem, answer))

    return problems_and_answers


# Generate the problems and answers
floating_point_problems_with_answers = generate_floating_point_problems_with_answers()

# Save the problems and answers to a text file
with open("floating_point_problems_with_answers.txt", "w") as f:
    for problem, answer in floating_point_problems_with_answers:
        f.write(f"Problem: {problem}, Answer: {answer}\n")

# Print a few examples for verification
for problem, answer in floating_point_problems_with_answers[:5]:
    print(f"Problem: {problem}, Answer: {answer}")
