import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse  # Import the argparse library for command-line argument parsing
from scipy import stats

"""
Explanation of Key Parts:

argparse Setup:
The argparse module is used to handle command-line arguments.
parser = argparse.ArgumentParser(...) creates an argument parser object.
parser.add_argument("-s", "--summarize_file", required=True, help="Path to the output_summarize_and_translate_D3.csv file.")
    # -s or --summarize_file: defines an argument for the summarize file path.
    # required=True: means the user must provide this argument.
    # help=\"...\": provides a description of the argument.
parser.add_argument("-t", "--translate_file", required=True, help="Path to the output_translate_and_summarize_D3.csv file.")
    # -t or --translate_file: defines an argument for the translate file path.
    # required=True: means the user must provide this argument.
    # help=\"...\": provides a description of the argument.
parser.add_argument("-o", "--output_dir", default="t-test", help="Path to the directory to save output files (default: t-test).")
    # -o or --output_dir: defines an argument for the output directory.
    # default=\"t-test\": sets the default output directory if not provided.
    # help=\"...\": provides a description of the argument.
args = parser.parse_args() parses the arguments provided by the user when running the script.

Input File Handling:
The script now directly uses the file paths provided by the user via the command-line arguments (args.summarize_file and args.translate_file).

Output Directory:
output_dir = args.output_dir uses the output directory specified by the user or the default "t-test".
if not os.path.exists(output_dir): os.makedirs(output_dir) checks if the directory exists and creates it if it doesn't.

Box Plot Generation:
The code iterates through each metric in the metrics list.
plt.figure(figsize=(8, 6)) creates a new figure for each box plot.
sns.boxplot(...) generates the box plot using the specified columns from the merged DataFrame.
plt.title(...), plt.xticks(...), plt.ylabel(...), and plt.xlabel(...) set the plot's title and labels.
plt.savefig(...) saves the plot as a PNG file in the output directory.
plt.close() closes the figure to release memory.

Formatted Output:
The results DataFrame is formatted using to_markdown() to create a clean, readable table.
This formatted table is then written to a text file using file.write().

User Feedback:
The script prints confirmation messages to the console, indicating where the results and box plots have been saved.

How to Run the Code:

Save the code: Save the code as a Python file (e.g., paired_t_test_script.py).
Open a terminal or command prompt: Navigate to the directory where you saved the Python file.
Run the script: Execute the script from the command line, providing the paths to the two CSV files and optionally the output directory. For example:
Bash

python paired_t_test_script.py -s path/to/output_summarize_and_translate_D3.csv -t path/to/output_translate_and_summarize_D3.csv -o my_analysis
Replace path/to/output_summarize_and_translate_D3.csv and path/to/output_translate_and_summarize_D3.csv with the actual paths to your files. The -o my_analysis part is optional; if you omit it, the results will be saved in a folder named t-test in the current directory.
If your files are named exactly as in the original script and are in a folder named data, and you want to save the output in a folder named analysis, you would run:
Bash

python paired_t_test_script.py -s data/output_summarize_and_translate_D3.csv -t data/output_translate_and_summarize_D3.csv -o analysis
"""

# 1. Set up the command-line argument parser.
parser = argparse.ArgumentParser(description="Perform paired t-tests and generate box plots for two input CSV files.")
parser.add_argument("-s", "--summarize_file", required=True, help="Path to the output_summarize_and_translate_D3.csv file.")
parser.add_argument("-t", "--translate_file", required=True, help="Path to the output_translate_and_summarize_D3.csv file.")
parser.add_argument("-o", "--output_dir", default="t-test", help="Path to the directory to save output files (default: t-test).")
args = parser.parse_args()

# 2. Create the output directory if it doesn't exist.
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)  # Create the directory if it doesn't exist

# 3. Read the CSV files into pandas DataFrames using the provided paths.
df_summarize = pd.read_csv(args.summarize_file)  # DataFrame for summarize_and_translate data
df_translate = pd.read_csv(args.translate_file)  # DataFrame for translate_and_summarize data

# 4. Merge the two DataFrames on the 'Filename' column.
merged_data = pd.merge(df_summarize, df_translate, on="Filename", suffixes=('_summarize', '_translate'))

# 5. Define the metrics to compare.
metrics = ["Total Processing Time", "ROUGE-1 F1", "ROUGE-L F1", "BERT F1"]
test_results = {}  # Dictionary to store the results of the t-tests

# 6. Perform paired t-tests for each metric.
for metric in metrics:
    # Calculate the difference between the metric values from the two methods.
    diff = merged_data[f"{metric}_summarize"] - merged_data[f"{metric}_translate"]
    # Perform the paired t-test using scipy.stats.ttest_rel.
    t_statistic, p_value = stats.ttest_rel(merged_data[f"{metric}_summarize"], merged_data[f"{metric}_translate"])
    # Store the t-statistic and p-value.
    test_results[metric] = {"t_statistic": t_statistic, "p_value": p_value}

# 7. Organize the test results into a DataFrame.
results_df = pd.DataFrame(test_results).T
results_df['significant'] = results_df['p_value'] < 0.05  # Mark significance based on p-value

# 8. Reorder the columns of the results DataFrame.
results_df = results_df[['t_statistic', 'p_value', 'significant']]

# 9. Rename the columns for better readability.
results_df.columns = ['t-statistic', 'p-value', 'significant']

# 10. Create and save box plots for each metric.
for metric in metrics:
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.boxplot(data=merged_data[[f"{metric}_summarize", f"{metric}_translate"]])  # Create box plot
    plt.title(f"Comparison of {metric}")  # Title of the plot
    plt.xticks([0, 1], ["summarize_and_translate", "translate_and_summarize"])  # X-axis labels
    plt.ylabel(metric)  # Y-axis label
    plt.xlabel("Method")  # X-axis label
    plt.savefig(os.path.join(args.output_dir, f"{metric}_box_plot.png"))  # Save plot to output directory
    plt.close()  # Close the figure to release memory

# 11. Save the formatted test results to a text file.
output_path = os.path.join(args.output_dir, "paired_t_test_results.txt")  # Output file path in specified directory
with open(output_path, "w", encoding="utf-8") as file:
    file.write("Paired t-test results:\n\n")  # Title for the output
    file.write(results_df.to_markdown(numalign="left", stralign="left"))  # Write DataFrame to file

print(f"\nResults saved to: {output_path}\n")  # Print output file path
print(f"Box plots saved to directory: {args.output_dir}/\n")  # Print directory message