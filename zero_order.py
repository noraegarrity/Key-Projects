import sys
import numpy as np

def f1(x1, x2):
    return x1**2 + x2**2 + 2

def f2(x1, x2):
    return x1**2 + 24 * x2**2

def f3(x1, x2):
    return x1**2 + 120 * x2**2

def f4(x1, x2):
    return x1**2 + 1200 * x2**2

functions = {"f1": f1, "f2": f2, "f3": f3, "f4": f4}

def random_search(f, w, alpha, N):
    print(0, w[0], w[1], f(*w))
    for k in range(1, N + 1):
        best_w = w
        best_fval = f(*w)
        
        for _ in range(10):
            x1 = np.random.uniform(-1, 1)
            x2 = np.sign(np.random.uniform(-1, 1)) * np.sqrt(1 - x1**2)
            directions = [(x1, x2), (-x1, -x2)]
            
            for d in directions:
                new_w = (w[0] + alpha * d[0], w[1] + alpha * d[1])
                fval = f(*new_w)
                if fval < best_fval:
                    best_w, best_fval = new_w, fval
        
        if best_fval < f(*w):
            w = best_w
            print(k, w[0], w[1], best_fval)
        else:
            print(k - 1, w[0], w[1], f(*w))

def coordinate_search(f, w, alpha, N):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    print(0, w[0], w[1], f(*w))
    for k in range(1, N + 1):
        best_w = w
        best_fval = f(*w)
        
        for d in directions:
            new_w = (w[0] + alpha * d[0], w[1] + alpha * d[1])
            fval = f(*new_w)
            if fval < best_fval:
                best_w, best_fval = new_w, fval
        
        if best_fval < f(*w):
            w = best_w
            print(k, w[0], w[1], best_fval)
        else:
            print(k - 1, w[0], w[1], f(*w))
            break

def coordinate_descent(f, w, alpha, N):
    prev_w = w
    prev_w2 = w
    for k in range(1, N + 1):
        axis = 0 if k % 2 == 1 else 1
        directions = [(-1, 0), (1, 0)] if axis == 0 else [(0, -1), (0, 1)]
        
        best_w = w
        best_fval = f(*w)
        
        for d in directions:
            new_w = (w[0] + alpha * d[0], w[1] + alpha * d[1])
            fval = f(*new_w)
            if fval < best_fval:
                best_w, best_fval = new_w, fval
        
        if best_fval < f(*w):
            w = best_w
            print(k, w[0], w[1], best_fval)
        else:
            print(k - 1, w[0], w[1], f(*w))
            if k > 2 and f(*w) >= f(*prev_w) and f(*w) >= f(*prev_w2):
                break
        prev_w2 = prev_w
        prev_w = w

def main():
    func_name, alpha, N, method_id, x1, x2, output_file = sys.argv[1:]
    alpha, N, method_id = float(alpha), int(N), int(method_id)
    x1, x2 = float(x1), float(x2)
    f = functions[func_name]
    
    w = (x1, x2)
    
    with open(output_file, "w") as f_out:
        sys.stdout = f_out
        
        if method_id == 1:
            random_search(f, w, alpha, N)
        elif method_id == 2:
            coordinate_search(f, w, alpha, N)
        elif method_id == 3:
            coordinate_descent(f, w, alpha, N)
        
        sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
