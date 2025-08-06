import sys
import math
import heapq
from collections import Counter, defaultdict

def parse_vector_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            label = tokens[0]
            vector = {word: int(count) for word, count in (item.split(':') for item in tokens[1:])}
            data.append((label, vector))
    return data

def compute_mag(data):
    return [math.sqrt(sum(v ** 2 for v in vector.values())) for _, vector in data]

def cosine_sim(vec1, vec2, mag1, mag2):
    if mag1 == 0 or mag2 == 0:
        return 0
    dot_product = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in vec1.keys() & vec2.keys())
    return dot_product / (mag1 * mag2)

def get_k_nn(train_data, test_vector, train_magnitudes, k, similarity_func):
    distances = []
    test_magnitude = math.sqrt(sum(v ** 2 for v in test_vector.values()))

    for idx, (train_label, train_vector) in enumerate(train_data):
        if similarity_func == 1:
            distance = sum((train_vector.get(key, 0) - test_vector.get(key, 0)) ** 2 for key in test_vector.keys())
            distance = math.sqrt(distance)
        else:
            similarity = cosine_sim(train_vector, test_vector, train_magnitudes[idx], test_magnitude)
            distance = 1 - similarity

        distances.append((train_label, distance))

    distances.sort(key=lambda x: x[1])
    return [label for label, _ in distances[:k]]

def predict_label(train_data, test_vector, k, similarity_func, train_magnitudes):
    neighbors = get_k_nn(train_data, test_vector, train_magnitudes, k, similarity_func)
    label_counts = Counter(neighbors)
    total = sum(label_counts.values())
    return sorted([(cls, count / total) for cls, count in label_counts.items()], key=lambda x: -x[1])

def classify(train_data, test_data, k, similarity_func, sys_output):
    train_magnitudes = compute_mag(train_data)

    with open(sys_output, 'w') as f:
        f.write("%%%%% training data:\n")
        train_predictions = []
        for idx, (true_label, vector) in enumerate(train_data):
            predicted_probs = predict_label(train_data, vector, k, similarity_func, train_magnitudes)
            f.write(f"array:{idx} {true_label} " + " ".join(f"{cls} {prob:.4f}" for cls, prob in predicted_probs) + "\n")
            train_predictions.append((true_label, predicted_probs[0][0]))

        f.write("%%%%% test data:\n")
        test_predictions = []
        for idx, (true_label, vector) in enumerate(test_data):
            predicted_probs = predict_label(train_data, vector, k, similarity_func, train_magnitudes)
            f.write(f"array:{idx} {true_label} " + " ".join(f"{cls} {prob:.4f}" for cls, prob in predicted_probs) + "\n")
            test_predictions.append((true_label, predicted_probs[0][0]))

    return train_predictions, test_predictions

def compute_accuracy(predictions, classes, acc_file, dataset_type):
    confusion_matrix = {cls: {cls_: 0 for cls_ in classes} for cls in classes}
    correct = sum(1 for true_label, predicted_label in predictions if true_label == predicted_label)

    for true_label, predicted_label in predictions:
        confusion_matrix[true_label][predicted_label] += 1

    accuracy = correct / len(predictions)

    with open(acc_file, 'a') as f:
        f.write(f"\nConfusion matrix for the {dataset_type} data:\n")
        f.write("row is the truth, column is the system output\n\n")
        f.write("             " + " ".join(classes) + "\n")
        for cls in classes:
            f.write(f"{cls} " + " ".join(str(confusion_matrix[cls][cls_]) for cls_ in classes) + "\n")
        f.write(f"\n {dataset_type.capitalize()} accuracy={accuracy:.15f}\n")

def main():
    train_file, test_file, k, similarity_func, sys_output, acc_file = sys.argv[1:]
    k = int(k)
    similarity_func = int(similarity_func)

    train_data = parse_vector_file(train_file)
    test_data = parse_vector_file(test_file)

    if k > len(train_data):
        k = len(train_data)
    if k < 2:
        k = 2

    classes = sorted(set(label for label, _ in train_data))

    with open(acc_file, 'w') as f:
        f.write("")

    train_predictions, test_predictions = classify(train_data, test_data, k, similarity_func, sys_output)
    compute_accuracy(train_predictions, classes, acc_file, "training")
    compute_accuracy(test_predictions, classes, acc_file, "test")

if __name__ == "__main__":
    main()
