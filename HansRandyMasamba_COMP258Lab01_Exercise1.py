import numpy as np

class DeltaRuleBase:
    
    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.zeros(input_size)
        self.bias = 0.0
        self.learning_rate = learning_rate


    def bipolar_conversion(self, pattern):
        # Convert vector to bipolar (-1,1) while flattening any nested structure
        flattened = np.asarray(pattern, dtype=float).ravel()
        return flattened - 1

    def predict(self, pattern):
        # Returns continuous output (ADALINE)
        return np.dot(self.weights, pattern) + self.bias

class Perceptron(DeltaRuleBase):

    def train(self, patterns, targets, epochs=100):
        for _ in range(epochs):
            for x, t in zip(patterns, targets):
                x_bipolar = self.bipolar_conversion(x)
                output = 1 if self.predict(x_bipolar) >= 0 else -1
                update = self.learning_rate * (t - output)
                self.weights += update * x_bipolar
                self.bias += update

        print(f"Perceptron Final Weights: {self.weights}")
        print(f"Perceptron Final Bias: {self.bias}")

    def classify(self, pattern):
        x_bipolar = self.bipolar_conversion(pattern)
        output = 1 if self.predict(x_bipolar) >= 0 else -1
        return output

class Adaline(DeltaRuleBase):

    def train(self, patterns, targets, epochs=100):
        for _ in range(epochs):
            for x, t in zip(patterns, targets):
                x_bipolar = self.bipolar_conversion(x)
                output = self.predict(x_bipolar)
                error = t - output
                self.weights += self.learning_rate * error * x_bipolar
                self.bias += self.learning_rate * error

        print(f"Adaline Final Weights: {self.weights}")
        print(f"Adaline Final Bias: {self.bias}")

    def classify(self, pattern):
        x_bipolar = self.bipolar_conversion(pattern)
        output = self.predict(x_bipolar)
        return 1 if output >= 0 else -1

def introduce_pixel_noise(pattern, count, rng):
    """Flip `count` pixels (0<->2). Intermediate values drift toward the opposite extreme."""
    arr = np.asarray(pattern, dtype=float).ravel().copy()
    count = max(0, min(count, arr.size))

    if count == 0:
        return arr.tolist()

    indices = rng.choice(arr.size, size=count, replace=False)

    for idx in indices:
        if arr[idx] == 0:
            arr[idx] = 2
        elif arr[idx] == 2:
            arr[idx] = 0
        else:
            arr[idx] = 2 if arr[idx] <= 1 else 0
    return arr.tolist()


def introduce_missing_data(pattern, count, rng):
    """Replace `count` pixels with the neutral mid-value (1).
    After bipolar conversion, these pixels contribute 0, modelling missing/unknown data.
    """
    arr = np.asarray(pattern, dtype=float).ravel().copy()
    count = max(0, min(count, arr.size))

    if count == 0:
        return arr.tolist()

    indices = rng.choice(arr.size, size=count, replace=False)
    arr[indices] = 1
    return arr.tolist()

def classify_with_activation(model, pattern):
    activation = model.predict(model.bipolar_conversion(pattern))
    label = "B" if activation >= 0 else "Not B"
    return label, activation

def display_comparison(perceptron, adaline, base_pattern, rng):
    scenarios = [("Original", base_pattern)]
    noise_levels = [5, 10, 15, 20]

    scenarios.extend(
        (f"{level} noisy px", introduce_pixel_noise(base_pattern, level, rng))
        for level in noise_levels
    )

    scenarios.extend(
        (f"{level} missing px", introduce_missing_data(base_pattern, level, rng))
        for level in noise_levels
    )

    header = "{:<18} | {:<20} | {:<20} | {}".format(
        "Scenario", "Perceptron", "Adaline", "Agreement"
    )


    print("Classification Results")
    print(header)
    print("-" * len(header))

    results = []

    for label, pattern in scenarios:
        perc_pred, perc_act = classify_with_activation(perceptron, pattern)
        ada_pred, ada_act = classify_with_activation(adaline, pattern)
        agree = perc_pred == ada_pred
        print(
            "{:<18} | {:<20} | {:<20} | {}".format(
                label,
                f"{perc_pred} ({perc_act:+.3f})",
                f"{ada_pred} ({ada_act:+.3f})",
                "Yes" if agree else "No",
            )
        )

        scenario_type = (
            "noise" if "noisy" in label else "missing" if "missing" in label else "baseline"
        )

        results.append(
            {
                "label": label,
                "type": scenario_type,
                "perceptron_pred": perc_pred,
                "perceptron_activation": perc_act,
                "adaline_pred": ada_pred,
                "adaline_activation": ada_act,
                "agreement": agree,
            }
        )

    def summarise(result_type, expected_label="B"):
        subset = [row for row in results if row["type"] == result_type]
        
        if not subset:
            return None

        perc_correct = sum(row["perceptron_pred"] == expected_label for row in subset)
        ada_correct = sum(row["adaline_pred"] == expected_label for row in subset)
        disagreements = [row["label"] for row in subset if not row["agreement"]]
        perc_margin = float(np.mean([row["perceptron_activation"] for row in subset]))
        ada_margin = float(np.mean([row["adaline_activation"] for row in subset]))

        return {
            "total": len(subset),
            "perc_correct": perc_correct,
            "ada_correct": ada_correct,
            "disagreements": disagreements,
            "perc_margin": perc_margin,
            "ada_margin": ada_margin,
        }



    noise_summary = summarise("noise")
    missing_summary = summarise("missing")

    print("Observations:")

    if noise_summary:
        print(
            f"- Noise: Perceptron kept the 'B' label {noise_summary['perc_correct']}/{noise_summary['total']} times (avg activation {noise_summary['perc_margin']:+.3f}); "
            f"Adaline {noise_summary['ada_correct']}/{noise_summary['total']} (avg activation {noise_summary['ada_margin']:+.3f})."
        )

        if noise_summary["disagreements"]:
            print(f"  Disagreement on: {', '.join(noise_summary['disagreements'])}.")
        if noise_summary["perc_correct"] > noise_summary["ada_correct"]:
            print("  Perceptron proved slightly more noise-tolerant in this run.")
        elif noise_summary["perc_correct"] < noise_summary["ada_correct"]:
            print("  Adaline proved slightly more noise-tolerant in this run.")
        else:
            print("  Both models handled noisy pixels similarly here.")



    if missing_summary:
        print(
            f"- Missing data: Perceptron kept the 'B' label {missing_summary['perc_correct']}/{missing_summary['total']} times (avg activation {missing_summary['perc_margin']:+.3f}); "
            f"Adaline {missing_summary['ada_correct']}/{missing_summary['total']} (avg activation {missing_summary['ada_margin']:+.3f})."
        )

        if missing_summary["disagreements"]:
            print(f"  Disagreement on: {', '.join(missing_summary['disagreements'])}.")
        if missing_summary["perc_correct"] > missing_summary["ada_correct"]:
            print("  Perceptron coped better with missing pixels in this run.")
        elif missing_summary["perc_correct"] < missing_summary["ada_correct"]:
            print("  Adaline coped better with missing pixels in this run.")
        else:
            print("  Both models reacted to missing pixels in the same way.")

    if not noise_summary and not missing_summary:
        print("- No additional scenarios were evaluated.")

if __name__ == "__main__":
    # Input patterns (7x9 pixels flattened to 63-length vectors)

    input_patterns = [
        # Font 1
        [0, 0, 2, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2, 0, 2, 2, 2], # A

        [2, 2, 2, 2, 2, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 2, 2, 2, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         2, 2, 2, 2, 2, 2, 0], # B

        [0, 0, 2, 2, 2, 2, 2,
         0, 2, 0, 0, 0, 0, 2,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         2, 0, 0, 0, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 0, 2, 2, 2, 2, 0], # C

        [2, 2, 2, 2, 2, 0, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 2, 0,
         2, 2, 2 ,2, 2, 0, 0], # D

        [2, 2, 2, 2, 2, 2, 2,
         0, 2, 0, 0, 0, 0, 2,
         0, 2, 0, 0, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 2, 2, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 0,
         0, 2, 0, 0, 0, 0, 2,
         2, 2, 2, 2, 2, 2, 2], # E

        [0, 0, 0, 2, 2, 2, 2,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 0, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 2, 0, 0, 0, 2, 0,
         0, 0, 2, 2, 2, 0, 0], # J

        [2, 2, 2, 0, 0, 2, 2,
         0, 2, 0, 0, 2, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 2, 0, 0, 0, 0,
         0, 2, 2, 0, 0, 0, 0,
         0, 2, 0, 2, 0, 0, 0,
         0, 2, 0, 0, 2, 0, 0, 
         0, 2, 0, 0, 0, 2, 0,
         2, 2, 2, 0, 0, 2, 2], # K

        # Font 2
        [0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0], # A

        [2, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0], # B

        [0, 0, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # C

        [2, 2, 2, 2, 2, 0, 0,   
         2, 0, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 2, 0,   
         2, 2, 2 ,2, 2, 0, 0], # D

        [2, 2, 2, 2, 2, 2, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 2, 2, 2, 2, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 2, 2, 2, 2, 2, 2], # E
        
        [0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # J

        [2, 0, 0, 0, 0, 2, 0,   
         2, 0, 0, 0, 2, 0, 0,   
         2, 0, 0, 2, 0, 0, 0,   
         2, 0, 2, 0, 0, 0, 0,   
         2, 2, 0, 0, 0, 0, 0,   
         2, 0, 2, 0, 0, 0, 0,   
         2, 0, 0, 2, 0, 0, 0,   
         2, 0, 0, 0, 2, 0, 0,   
         2, 0, 0, 0, 0, 2, 0], # K

        # Font 3
        [0, 0, 0, 2, 0, 0, 0,   
         0, 0, 0, 2, 0, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 0, 2, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 2, 2, 2, 2, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 2, 0, 0, 0, 2 ,2], # A

        [2, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 2, 2, 2, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 0], # B
        
        [0, 0, 2, 2, 2, 0, 2,   
         0, 2, 0, 0, 0, 2, 2,   
         2, 0, 0, 0, 0, 0, 2,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 0,   
         2, 0, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # C

        [2, 2, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2 ,2, 2, 0, 0], # D
       
        [2, 2, 2, 2, 2, 2, 2,   
         0, 2, 0, 0, 0, 0, 2,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 2, 2, 2, 0, 0,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 0, 0,   
         0, 2, 0, 0, 0, 0, 0,   
         0, 2, 0, 0, 0, 0, 2,   
         2, 2, 2, 2, 2, 2, 2], # E

        [0, 0, 0, 0, 2, 2, 2,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 0, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 0, 2, 2, 2, 0, 0], # J

        [2, 2, 2, 0, 0, 2, 2,   
         0, 2, 0, 0, 0, 2, 0,   
         0, 2, 0, 0, 2, 0, 0,   
         0, 2, 0, 2, 0, 0, 0,   
         0, 2, 2, 0, 0, 0, 0,  
         0, 2, 0, 2, 0, 0, 0,   
         0 ,2, 0, 0, 2, 0, 0,   
         0, 2, 0, 0, 0, 2, 0,   
         2, 2, 2, 0, 0, 2, 2], # K
    ]

    # For "B vs not B" classification, targets are 1 for B, -1 otherwise)
    targets = [
        -1, 1, -1, -1, -1, -1, -1,  # Font 1: Only "B" is 1
        -1, 1, -1, -1, -1, -1, -1,  # Font 2
        -1, 1, -1, -1, -1, -1, -1,  # Font 3
    ]

    perceptron = Perceptron(input_size=63, learning_rate=0.01)
    perceptron.train(input_patterns, targets)

    adaline = Adaline(input_size=63, learning_rate=0.01)
    adaline.train(input_patterns, targets)

    # Evaluate the trained models on variations of the Font 1 "B"
    base_pattern = input_patterns[1][:]
    rng = np.random.default_rng(42)
    display_comparison(perceptron, adaline, base_pattern, rng)