import sys
import numpy as np
import numdifftools as nd

def f1(x1, x2):
    return x1**2 + x2**2 + 2

def f2(x1, x2):
    return x1**2 + 24 * x2**2

def f3(x1, x2):
    return x1**2 + 120 * x2**2

def f4(x1, x2):
    return x1**2 + 1200 * x2**2

def g1(x):
    return np.sin(3*x)

def g2(x):
    return np.sin(3*x) + 0.1*x**2

def g3(x):
    return x**2 + 0.2

def g4(x):
    return x**3

def g5(x):
    return (x**4 + x**2 + 10*x) / 50

def g6(x):
    term1 = np.clip((x - 2.3)**3 + 1, -1e8, 1e8)
    term2 = np.clip((-3*x + 0.7)**3 + 1, -1e8, 1e8)
    return np.clip(term1**2 + term2**2, -1e8, 1e8)

functions = {"f1": f1, "f2": f2, "f3": f3, "f4": f4,
             "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g6": g6}

def gradient_descent(f, w, alpha, N, output_file):
    grad = nd.Gradient(lambda w: f(*w))
    history = [(0, *w, f(*w))]

    for k in range(1, N + 1):
        d = grad(w)

        d = np.atleast_1d(d)

        if len(d) != len(w):
            raise ValueError(f"Gradient size mismatch: Expected {len(w)}, but got {len(d)}")

        w = tuple(w[i] - alpha * d[i] for i in range(len(w)))
        history.append((k, *w, f(*w)))

    m, epsilon = 5, 0.001
    converged = all(
        np.linalg.norm(np.array(history[i][1:-1]) - np.array(history[i-1][1:-1])) < epsilon and
        abs(history[i][-1] - history[i-1][-1]) < epsilon
        for i in range(-m, 0)
    ) if len(history) > m else False

    last_w, last_f = history[-1][1:-1], history[-1][-1]
    if converged:
        result = "yes" if np.linalg.norm(last_w) < 1e8 and abs(last_f) < 1e8 else "yes-but-diverge"
    else:
        result = "no"

    with open(output_file, "w") as f_out:
        for line in history:
            f_out.write(" ".join(map(str, line)) + "\n")
        f_out.write(result + "\n")

def main():
    args = sys.argv[1:]
    func_name, alpha, N = args[:3]
    alpha, N = float(alpha), int(N)

    if func_name in {"f1", "f2", "f3", "f4"}:
        x1, x2 = map(float, args[3:5])
        output_file = args[5]
        w = (x1, x2)
    else:
        x1 = float(args[3])
        output_file = args[4]
        w = (x1,)

    f = functions[func_name]
    gradient_descent(f, w, alpha, N, output_file)

if __name__ == "__main__":
    main()