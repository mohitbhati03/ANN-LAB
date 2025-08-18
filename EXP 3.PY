def perceptron_logic(gate, inputs):
    n = len(inputs)

    if gate == "AND":
        w = [1] * n
        t = n

    elif gate == "OR":
        w = [1] * n
        t = 1

    elif gate == "NAND":
        w = [1] * n
        t = n

    elif gate == "NOR":
        w = [1] * n
        t = 1

    elif gate == "NOT":
        if n != 1:
            return "NOT gate only accepts 1 input."
        w = [-1]
        t = 0

    else:
        return "Invalid gate selected."

    summation = sum([i * j for i, j in zip(w, inputs)])

    if gate == "NAND" or gate == "NOR":
        output = 0 if summation >= t else 1
    else:
        output = 1 if summation >= t else 0

    return output

while True:
    print("\nAvailable Gates: AND, OR, NOT, NAND, NOR")
    gate = input("Enter logic gate: ").strip().upper()

    try:
        num_inputs = int(input("\nEnter number of inputs: "))
        if gate == "NOT" and num_inputs != 1:
            print("NOT gate only accepts 1 input.")
            continue
    except ValueError:
        print("Invalid number!")
        continue

    inputs = []
    for i in range(num_inputs):
        while True:
            try:
                x = int(input(f"Enter input {i+1} (0 or 1): "))
                if x not in [0, 1]:
                    print("Only 0 or 1 allowed.")
                    continue
                inputs.append(x)
                break
            except ValueError:
                print("Enter valid integer (0 or 1).")

    result = perceptron_logic(gate, inputs)
    print(f"{gate} Gate: {inputs} ⇒ {result}")

    again = input("\nDo you want to try again? (yes/no): ").strip().lower()
    if again != "yes":
        print("Program exited.")
        break
