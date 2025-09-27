import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Since there's only one iteration, plot the single evaluation result
iterations = [0]
eval_returns = [0.14122484624385834]  # From the console output

plt.figure(figsize=(10, 6))
plt.plot(iterations, eval_returns, marker="o", label="Eval Average Return")
plt.xlabel("Iteration")
plt.ylabel("Average Return")
plt.title("Behavior Cloning: Evaluation Average Return on Walker2d-v5")
plt.legend()
plt.grid(True)
plt.savefig("bc_eval_return.png")
print("Plot saved as bc_eval_return.png")
plt.close()
