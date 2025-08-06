import sys
import math
from collections import defaultdict, Counter

def load_data(file_path):
    data = []
    vocabulary = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = parts[0]
            features = Counter(word.split(':')[0] for word in parts[1:])
            vocabulary.update(features.keys())
            data.append((label, features))
    return data, sorted(vocabulary)

def train_nb(data, vocab, class_prior_delta, cond_prob_delta):
    class_counts = Counter()
    feature_counts = defaultdict(lambda: Counter())
    total_docs = len(data)

    for label, features in data:
        class_counts[label] += 1
        for feature in features:
            feature_counts[label][feature] += 1

    class_priors = {
        label: (count + class_prior_delta) / (total_docs + class_prior_delta * len(class_counts))
        for label, count in class_counts.items()
    }

    cond_probs = defaultdict(dict)
    for label, counts in feature_counts.items():
        total_features = sum(counts.values()) + cond_prob_delta * len(vocab)
        for feature in vocab:
            prob = (counts[feature] + cond_prob_delta) / total_features
            cond_probs[label][feature] = prob

    return class_priors, cond_probs

def classify(instance, vocab, class_priors, cond_probs):
    scores = {}
    for label in class_priors:
        log_prob = math.log10(class_priors[label])
        for feature in vocab:
            if feature in instance:
                log_prob += math.log10(cond_probs[label].get(feature, 1e-10))
        scores[label] = log_prob
    return scores

def predict(data, vocab, class_priors, cond_probs):
    predictions = []
    for label, features in data:
        scores = classify(features, vocab, class_priors, cond_probs)
        predicted_label = max(scores, key=scores.get)
        predictions.append((label, predicted_label, scores))
    return predictions

def write_model_file(model_file, class_priors, cond_probs):
    with open(model_file, 'w') as f:
        f.write("%%%%% prior prob P(c) %%%%%\n")
        for label, prob in class_priors.items():
            f.write(f"{label} {prob:.10f} {math.log10(prob):.10f}\n")
        f.write("%%%%% conditional prob P(f|c) %%%%%\n")
        for label, features in cond_probs.items():
            f.write(f"%%%%% conditional prob P(f|c) c={label} %%%%%\n")
            for feature, prob in features.items():
                f.write(f"{feature} {label} {prob:.10f} {math.log10(prob):.10f}\n")

def write_sys_output(sys_output, predictions, data_type):
    with open(sys_output, 'a') as f:
        f.write(f"%%%%% {data_type} data:\n")
        for i, (true_label, predicted_label, scores) in enumerate(predictions):
            scores_str = " ".join(f"{label} {10 ** score:.4f}" for label, score in sorted(scores.items(), key=lambda x: x[1], reverse=True))
            f.write(f"array:{i} {true_label} {predicted_label} {scores_str}\n")

def calc_accuracy(predictions):
    correct = sum(1 for true, pred, _ in predictions if true == pred)
    return correct / len(predictions)

def write_acc_file(acc_file, train_predictions, test_predictions):
    labels = sorted(set(true for true, _, _ in train_predictions + test_predictions))
    
    train_confusion = {true: Counter() for true in labels}
    for true, pred, _ in train_predictions:
        train_confusion[true][pred] += 1
    
    test_confusion = {true: Counter() for true in labels}
    for true, pred, _ in test_predictions:
        test_confusion[true][pred] += 1

    with open(acc_file, 'w') as f:
        f.write("Confusion matrix for the training data:\n")
        f.write("row is the truth, column is the system output\n\n")
        f.write("\t" + "\t".join(labels) + "\n")
        for label in labels:
            row = "\t".join(str(train_confusion[label][pred]) for pred in labels)
            f.write(f"{label}\t{row}\n")
        
        train_accuracy = calc_accuracy(train_predictions)
        f.write(f"\n Training accuracy={train_accuracy:.15f}\n\n")
        
        f.write("Confusion matrix for the test data:\n")
        f.write("row is the truth, column is the system output\n\n")
        f.write("\t" + "\t".join(labels) + "\n")
        for label in labels:
            row = "\t".join(str(test_confusion[label][pred]) for pred in labels)
            f.write(f"{label}\t{row}\n")
        
        test_accuracy = calc_accuracy(test_predictions)
        f.write(f"\n Test accuracy={test_accuracy:.15f}\n\n")

def main():
    train_file, test_file, class_prior_delta, cond_prob_delta, model_file, sys_output, acc_file = sys.argv[1:]
    class_prior_delta = float(class_prior_delta)
    cond_prob_delta = float(cond_prob_delta)

    train_data, vocab = load_data(train_file)
    test_data, _ = load_data(test_file)

    class_priors, cond_probs = train_nb(train_data, vocab, class_prior_delta, cond_prob_delta)

    write_model_file(model_file, class_priors, cond_probs)

    train_predictions = predict(train_data, vocab, class_priors, cond_probs)
    test_predictions = predict(test_data, vocab, class_priors, cond_probs)
    
    write_sys_output(sys_output, train_predictions, "training")
    write_sys_output(sys_output, test_predictions, "test")
    write_acc_file(acc_file, train_predictions, test_predictions)

if __name__ == "__main__":
    main()
