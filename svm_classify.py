import sys
import numpy as np
import math

def parse_libsvm_data(file_path):
    """Parses a libSVM format data file."""
    data = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            features = {int(kv.split(':')[0]): float(kv.split(':')[1]) for kv in parts[1:]}
            data.append(features)
    return labels, data

def parse_model_file(model_file):
    """Parses the model file and extracts necessary parameters."""
    with open(model_file, 'r') as f:
        lines = f.readlines()
    
    model = {}
    support_vectors = []
    alpha_y = []
    
    parsing_sv = False
    for line in lines:
        line = line.strip()
        if line.startswith("svm_type"):
            model['svm_type'] = line.split()[1]
        elif line.startswith("kernel_type"):
            model['kernel_type'] = line.split()[1]
        elif line.startswith("gamma"):
            model['gamma'] = float(line.split()[1])
        elif line.startswith("coef0"):
            model['coef0'] = float(line.split()[1])
        elif line.startswith("degree"):
            model['degree'] = int(line.split()[1])
        elif line.startswith("rho"):
            model['rho'] = float(line.split()[1])
        elif line == "SV":
            parsing_sv = True
        elif parsing_sv:
            parts = line.split()
            alpha_y.append(float(parts[0]))
            features = {int(kv.split(':')[0]): float(kv.split(':')[1]) for kv in parts[1:]}
            support_vectors.append(features)
    
    model['support_vectors'] = support_vectors
    model['alpha_y'] = np.array(alpha_y)
    return model

def compute_kernel(kernel_type, x1, x2, model):
    """Computes the kernel function between two sparse vectors."""
    gamma = model.get('gamma', 1)
    coef0 = model.get('coef0', 0)
    degree = model.get('degree', 3)

    if kernel_type == "linear":
        return sum(x1.get(k, 0) * x2.get(k, 0) for k in set(x1) | set(x2))
    elif kernel_type == "polynomial":
        return (gamma * sum(x1.get(k, 0) * x2.get(k, 0) for k in set(x1) | set(x2)) + coef0) ** degree
    elif kernel_type == "rbf":
        diff = sum((x1.get(k, 0) - x2.get(k, 0)) ** 2 for k in set(x1) | set(x2))
        return math.exp(-gamma * diff)
    elif kernel_type == "sigmoid":
        return math.tanh(gamma * sum(x1.get(k, 0) * x2.get(k, 0) for k in set(x1) | set(x2)) + coef0)
    return 0

def classify(test_data, model):
    """Classifies test instances based on the trained SVM model."""
    kernel_type = model['kernel_type']
    support_vectors = model['support_vectors']
    alpha_y = model['alpha_y']
    rho = model['rho']
    
    predictions = []
    for test_vector in test_data:
        fx = sum(alpha_y[i] * compute_kernel(kernel_type, test_vector, sv, model) for i, sv in enumerate(support_vectors)) - rho
        sys_label = 0 if fx >= 0 else 1
        predictions.append((sys_label, fx))
    
    return predictions

def main():
    if len(sys.argv) != 4:
        print("Usage: svm_classify.py test_data model_file sys_output_prefix")
        sys.exit(1)
    
    test_data_file = sys.argv[1]
    model_file = sys.argv[2]
    sys_output_prefix = sys.argv[3]
    
    true_labels, test_data = parse_libsvm_data(test_data_file)
    model = parse_model_file(model_file)
    
    experiment_mappings = {
        "linear": 1,
        "polynomial_1_0_2": 2,
        "polynomial_0.1_0.5_2": 3,
        "rbf_0.5": 4,
        "sigmoid_0.5_-0.2": 5
    }
    
    print(f"Kernel Type: {model['kernel_type']}")
    print(f"Gamma: {model.get('gamma', 'N/A')}")
    print(f"Coef0: {model.get('coef0', 'N/A')}")
    print(f"Degree: {model.get('degree', 'N/A')}")

    kernel_key = model['kernel_type']
    if model['kernel_type'] == "polynomial":
        kernel_key += f"_{model.get('gamma', '-')}" \
                      f"_{model.get('coef0', '-')}" \
                      f"_{model.get('degree', '-')}"
    elif model['kernel_type'] in ["rbf", "sigmoid"]:
        kernel_key += f"_{model.get('gamma', '-')}"
        if model['kernel_type'] == "sigmoid":
            kernel_key += f"_{model.get('coef0', '-')}"
    
    exp_id = experiment_mappings.get(kernel_key, 0)
    if exp_id == 0:
        print(f"Warning: Experiment ID not found for kernel configuration {kernel_key}.")
    
    sys_output_file = f"{sys_output_prefix}.{exp_id}"
    predictions = classify(test_data, model)
    
    with open(sys_output_file, 'w') as f:
        for true_label, (sys_label, fx) in zip(true_labels, predictions):
            f.write(f"{true_label} {sys_label} {fx}\n")

if __name__ == "__main__":
    main()
