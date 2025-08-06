# How to execute this .py file:
# 1. Save the file: Save the above code as a .py file, for example, analyze_data.py.
# 2. Open the terminal or command prompt: Navigate to the directory where you saved analyze_data.py.
# 3. Execute the command: Use the python command to run the script, using the -i parameter to specify the path to your CSV file, and the -o parameter to specify the folder path where you want to save the output files and images.
#
#    For example, if your CSV file is named output_summarize_and_translate_D3.csv, and you want to save the output in a folder named stats_summarize_and_translate_D3, you can execute the following command:
#
#    python analyze_data.py -i output_summarize_and_translate_D3.csv -o stats_summarize_and_translate_D3
#
#    If the -o parameter is omitted, the program will create a folder named output_plots in the current working directory to save the images.
#
# Files and charts generated after program execution, and their meanings:
# 1. analysis_output.txt file (located in the folder specified by -o):
#
#      Meaning: This file contains all the text output by the program in the terminal, including:
#          Messages indicating successful reading of the CSV file.
#          The first few rows of the CSV file (df.head()), allowing you to quickly view the data structure.
#          A summary of the data information (df.info()), including column names, the number of non-null values, and data types.
#          Descriptive statistics (df.describe()), including the count, mean, standard deviation, minimum, quartiles, and maximum for each numerical column, helping you understand the overall distribution and range of the data.
#          The correlation coefficient matrix (correlation_matrix), showing the linear correlation between different evaluation metrics and processing time.
#          Skewness and Kurtosis values: Quantify the asymmetry and "peakedness" of the data distribution for key metrics like ROUGE-1 F1, BERT F1, and ROUGE-L F1.
#          Outlier counts: Identifies the number of data points considered outliers (typically outside 1.5 times the Interquartile Range) for relevant numerical columns.
#          Spearman correlation matrix: Shows the monotonic relationships (not necessarily linear) between the key evaluation metrics and processing times. An optional Kendall correlation matrix can also be enabled.
#
# 2. PNG image files (located in the folder specified by -o):
#
#      ROUGE-1_f1_histogram.png and BERT_f1_histogram.png (histograms):
#          Meaning: Show the distribution of ROUGE-1 F1 and BERT F1 scores. The x-axis of the chart is the score value, and the y-axis is the frequency (or density). The kde=True overlaid kernel density estimation line can help you observe the shape of the distribution more smoothly (e.g., whether it is close to a normal distribution, whether it is skewed).
#      summarization_time_boxplot.png and original_char_count_boxplot.png (box plots):
#          Meaning: Show the distribution, median (the middle line), quartiles (the upper and lower boundaries of the box), and potential outliers (points outside the whiskers) of "summarization time" and "original Chinese text character count". The length of the box represents the interquartile range (IQR), and the whiskers usually extend to 1.5 times the IQR.
#      ROUGE-1_vs_BERT_scatter.png (scatter plot):
#          Meaning: Shows the relationship between ROUGE-1 F1 scores and BERT F1 scores. Each point represents a data sample. You can observe whether these two different evaluation metrics show a certain trend (e.g., whether a high ROUGE score corresponds to a high BERTScore).
#      char_count_vs_rouge1_scatter.png (scatter plot):
#          Meaning: Shows the relationship between the "original Chinese text character count" and ROUGE-1 F1 scores, helping you understand whether the length of the original text might affect the ROUGE evaluation results of the summary.
#      correlation_matrix_heatmap.png (heatmap):
#          Meaning: Displays the correlation matrix between different evaluation metrics and processing time using color intensity. The darker the color, the stronger the correlation (positive or negative), and the lighter the color, the weaker the correlation. The numbers on the chart are the specific correlation coefficient values. This can help you quickly identify which variables have a strong linear relationship.
#      ROUGE-L_f1_histogram.png (histogram): Displays the distribution of ROUGE-L F1 scores.
#      ROUGE-1_f1_boxplot.png, BERT_f1_boxplot.png, ROUGE-L_f1_boxplot.png (box plots): Provide a clear view of the median, quartiles, and outliers for ROUGE-1 F1, BERT F1, and ROUGE-L F1 scores.
#      ROUGE-1_f1_violinplot.png, BERT_f1_violinplot.png, ROUGE-L_f1_violinplot.png (violin plots): Combine box plots and kernel density estimates to display the full distribution shape and density for ROUGE-1 F1, BERT F1, and ROUGE-L F1 scores.
#      ROUGE-1_f1_cdfplot.png, BERT_f1_cdfplot.png, ROUGE-L_f1_cdfplot.png (cumulative distribution function plots): Illustrate the proportion of data points below a certain value for ROUGE-1 F1, BERT F1, and ROUGE-L F1 scores, useful for understanding percentiles.
#      ROUGE-1_f1_qqplot.png, BERT_f1_qqplot.png, ROUGE-L_f1_qqplot.png (QQ plots / Quantile-Quantile Plots): Assess whether ROUGE-1 F1, BERT F1, and ROUGE-L F1 scores follow a normal distribution.
#      key_metrics_pairplot.png (pair plot/scatter matrix plot): Displays pairwise scatter relationships between key numerical metrics and their individual distributions on the diagonal.
#      {metric_name_formatted}_by_model_type_boxplot.png, {metric_name_formatted}_by_model_type_violinplot.png, rouge1_vs_bert_by_model_type_scatter.png (grouped charts): If the data includes a 'model_type' column, these charts further categorize and display the distribution or relationship of metrics by model type.
#
# Suggestions for further analysis:
#
# After completing these basic descriptive statistics and visualizations, you can conduct more in-depth analysis based on your research questions and data characteristics:
#
# 1. More detailed univariate analysis:
#      Calculate skewness and kurtosis to describe the shape of the data distribution more accurately.
#      Identify and handle outliers, for example, using the IQR method or other statistical methods.
#      If your data includes different models or settings, you can plot histograms and box plots for different groups for comparison.
#
# 2. More in-depth bivariate analysis:
#      Calculate Spearman or Kendall correlation coefficients between different evaluation metrics to assess non-linear relationships.
#      If your data includes categorical variables (e.g., model type), you can plot grouped scatter plots or box plots to compare the performance of different groups on the evaluation metrics.
#
# 3. Multivariate analysis:
#      If there are multiple predictor variables (e.g., original text length, summary length, processing time), you can try building linear regression models to predict the scores of the evaluation metrics.
#      Use dimensionality reduction techniques (e.g., Principal Component Analysis, PCA) to explore the underlying structure in the data.
#
# 4. Hypothesis testing:
#      If you want to compare whether there is a statistically significant difference in the average performance of different models or settings on a specific evaluation metric, you can use hypothesis testing methods such as t-tests and ANOVA.
#
# 5. Time series analysis (if applicable):
#      If your data is collected sequentially over time, you can perform time series analysis to explore trends and patterns.
#
# 6. Integration with domain knowledge:
#      Combine your statistical analysis results with your domain knowledge of text summarization and evaluation metrics to interpret your findings and draw meaningful conclusions.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
import statsmodels.api as sm # Required for QQ Plot (Quantile-Quantile Plot)

# Define a Tee class to simultaneously write standard output to the console and a file
class Tee(object):
    def __init__(self, name, mode):
        # Save the standard output (console) to self.stdout
        self.stdout = sys.stdout
        # Open the file in the specified mode ('w' for write, 'a' for append)
        self.file = open(name, mode)

    # Override the write method so that written data goes to both console and file
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        # Force the buffer to be written to the file
        self.file.flush()

    # Override the flush method to ensure both console and file buffers are flushed
    def flush(self):
        self.stdout.flush()
        self.file.flush()

if __name__ == "__main__":
    # Create an ArgumentParser object to handle command-line arguments
    parser = argparse.ArgumentParser(description="Analyze summary evaluation metrics from a CSV file.")
    # Add a required command-line argument -i or --input_csv for the input CSV file path
    parser.add_argument("-i", "--input_csv", required=True, help="Path to the input CSV file.")
    # Add an optional command-line argument -o or --output_dir for the output directory of plots, default is './output_plots'
    parser.add_argument("-o", "--output_dir", default="./output_plots", help="Path to the directory for saving output plots (default: ./output_plots).")

    # Parse the command-line arguments
    args = parser.parse_args()
    input_csv_file = args.input_csv
    output_directory = args.output_dir

    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Define the path for the output log file within the specified output directory
    output_log_file = os.path.join(output_directory, 'analysis_output.txt')
    # Redirect standard output to the Tee object, so print output goes to both console and log file
    sys.stdout = Tee(output_log_file, 'w')

    try:
        # 1. Read CSV File
        df = pd.read_csv(input_csv_file)
        print(f"--- Successfully read data from: {input_csv_file} ---")
        print("--- First few rows of data ---")
        print(df.head())
        print("\n--- Data Information ---")
        print(df.info())

        # 2. Descriptive Statistics
        print("\n--- Descriptive Statistics ---")
        pd.set_option('display.max_columns', None)  # <-- Add this line
        print(df.describe())

        # --- Statistical Value Analyses ---
        # 2.1 Skewness and Kurtosis
        print("\n--- Skewness and Kurtosis ---")
        metrics_for_skew_kurt = ['ROUGE-1 F1', 'BERT F1']
        if 'ROUGE-L F1' in df.columns:
            metrics_for_skew_kurt.append('ROUGE-L F1')

        for col in metrics_for_skew_kurt:
            if col in df.columns:
                print(f"  {col} - Skewness: {df[col].skew():.4f}")
                print(f"  {col} - Kurtosis: {df[col].kurt():.4f}")

        # 2.2 Outlier Identification
        print("\n--- Outlier Counts (using 1.5*IQR rule) ---")
        numerical_cols_for_outliers = ['ROUGE-1 F1', 'BERT F1', 'Summarization Time', 'Original Chinese Character Count']
        if 'ROUGE-L F1' in df.columns:
            numerical_cols_for_outliers.append('ROUGE-L F1')

        for col in numerical_cols_for_outliers:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + 1.5 * IQR
                lower_bound = Q1 - 1.5 * IQR
                outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
                print(f"  {col}: {outliers_count} outliers found.")

        # 2.3 Spearman or Kendall Correlation
        print("\n--- Spearman Correlation Matrix ---")
        cols_for_corr = ['ROUGE-1 F1', 'BERT F1', 'Original Chinese Character Count', 'Summary English Word Count', 'Summarization Time', 'Translation Time']
        if 'ROUGE-L F1' in df.columns:
            cols_for_corr.append('ROUGE-L F1')

        # Ensure all columns exist before calculating correlation
        existing_cols_for_corr = [col for col in cols_for_corr if col in df.columns]
        if existing_cols_for_corr:
            spearman_correlation_matrix = df[existing_cols_for_corr].corr(method='spearman')
            print(spearman_correlation_matrix)
        else:
            print("No relevant numerical columns found for Spearman correlation.")

        # Optional: Kendall Correlation Matrix (uncomment to enable)
        # print("\n--- Kendall Correlation Matrix ---")
        # if existing_cols_for_corr:
        #     kendall_correlation_matrix = df[existing_cols_for_corr].corr(method='kendall')
        #     print(kendall_correlation_matrix)
        # else:
        #     print("No relevant numerical columns found for Kendall correlation.")
        # --- End Statistical Value Analyses ---


        # 3. Univariate Analysis (Existing)
        # Helper function to get standardized filename base
        def get_filename_base(metric_name):
            if metric_name == 'ROUGE-1 F1':
                return 'ROUGE-1_f1'
            elif metric_name == 'BERT F1':
                return 'BERT_f1'
            elif metric_name == 'ROUGE-L F1':
                return 'ROUGE-L_f1'
            return metric_name.lower().replace(" ", "_") # Fallback for other metrics

        # 3.1 Histogram - Display the distribution of a single variable
        plt.figure(figsize=(10, 6))
        sns.histplot(df['ROUGE-1 F1'], bins=30, kde=True)
        plt.title('Distribution of ROUGE-1 F1')
        plt.xlabel('ROUGE-1 F1 Score')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_directory, 'ROUGE-1_f1_histogram.png'))
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(df['BERT F1'], bins=30, kde=True)
        plt.title('Distribution of BERT F1')
        plt.xlabel('BERT F1 Score')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_directory, 'BERT_f1_histogram.png'))
        plt.show()

        # ROUGE-L F1 Histogram
        if 'ROUGE-L F1' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['ROUGE-L F1'], bins=30, kde=True)
            plt.title('Distribution of ROUGE-L F1')
            plt.xlabel('ROUGE-L F1 Score')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_directory, 'ROUGE-L_f1_histogram.png'))
            plt.show()

        # 3.2 Box Plot - Display the distribution, median, quartiles, and potential outliers of a single variable
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df['Summarization Time'])
        plt.title('Box Plot of Summarization Time')
        plt.ylabel('Summarization Time (seconds)')
        plt.savefig(os.path.join(output_directory, 'summarization_time_boxplot.png'))
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df['Original Chinese Character Count'])
        plt.title('Box Plot of Original Chinese Character Count')
        plt.ylabel('Original Character Count')
        plt.savefig(os.path.join(output_directory, 'original_char_count_boxplot.png'))
        plt.show()

        # Additional Univariate Plots for ROUGE/BERT/ROUGE-L
        print("\n--- Generating Additional Univariate Plots for Metrics ---")
        for metric in metrics_for_skew_kurt: # Uses the list defined earlier
            if metric in df.columns:
                filename_base = get_filename_base(metric)

                # Box Plot
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df[metric])
                plt.title(f'Box Plot of {metric}')
                plt.ylabel(f'{metric} Score')
                plt.savefig(os.path.join(output_directory, f'{filename_base}_boxplot.png'))
                plt.show()

                # Violin Plot
                plt.figure(figsize=(10, 6))
                sns.violinplot(y=df[metric])
                plt.title(f'Violin Plot of {metric}')
                plt.ylabel(f'{metric} Score')
                plt.savefig(os.path.join(output_directory, f'{filename_base}_violinplot.png'))
                plt.show()

                # Cumulative Distribution Function (CDF) Plot
                plt.figure(figsize=(10, 6))
                sns.ecdfplot(data=df, x=metric)
                plt.title(f'Cumulative Distribution Function of {metric}')
                plt.xlabel(f'{metric} Score')
                plt.ylabel('Cumulative Probability')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(os.path.join(output_directory, f'{filename_base}_cdfplot.png'))
                plt.show()

                # QQ Plot (Quantile-Quantile Plot) - Requires 'statsmodels'
                try:
                    plt.figure(figsize=(8, 8))
                    sm.qqplot(df[metric].dropna(), line='s', fit=True) # dropna() to handle missing values
                    plt.title(f'QQ Plot of {metric}')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.savefig(os.path.join(output_directory, f'{filename_base}_qqplot.png'))
                    plt.show()
                except Exception as e:
                    print(f"Warning: Could not generate QQ Plot for {metric}: {e}")
                    print("Please ensure 'statsmodels' is installed (pip install statsmodels) and data is suitable.")
        # --- End Additional Univariate Plots ---


        # 4. Bivariate Analysis (Existing)
        # 4.1 Scatter Plot - Display the relationship between two variables
        plt.figure(figsize=(8, 6))
        plt.scatter(df['ROUGE-1 F1'], df['BERT F1'])
        plt.title('ROUGE-1 F1 vs. BERT F1')
        plt.xlabel('ROUGE-1 F1 Score')
        plt.ylabel('BERT F1 Score')
        plt.savefig(os.path.join(output_directory, 'ROUGE-1_vs_BERT_scatter.png')) # Updated filename
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(df['Original Chinese Character Count'], df['ROUGE-1 F1'])
        plt.title('Original Chinese Character Count vs. ROUGE-1 F1')
        plt.xlabel('Original Chinese Character Count')
        plt.ylabel('ROUGE-1 F1 Score')
        plt.savefig(os.path.join(output_directory, 'char_count_vs_rouge1_scatter.png')) # Filename remains consistent with original
        plt.show()

        # Grouped Bivariate Plots (if categorical column exists)
        # This section assumes your CSV has a categorical column, e.g., 'model_type'
        # Please replace 'model_type' with your actual categorical column name if different.

        if 'model_type' in df.columns: # Check if the categorical column exists
            print("\n--- Generating Grouped Plots by Model Type ---")
            for metric in metrics_for_skew_kurt: # Uses the list defined earlier
                if metric in df.columns:
                    filename_base = get_filename_base(metric)
                    # Conditional Box Plot
                    plt.figure(figsize=(12, 7))
                    sns.boxplot(x='model_type', y=metric, data=df)
                    plt.title(f'{metric} by Model Type')
                    plt.xlabel('Model Type')
                    plt.ylabel(f'{metric} Score')
                    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_directory, f'{filename_base}_by_model_type_boxplot.png'))
                    plt.show()

                    # Conditional Violin Plot
                    plt.figure(figsize=(12, 7))
                    sns.violinplot(x='model_type', y=metric, data=df)
                    plt.title(f'{metric} by Model Type (Violin Plot)')
                    plt.xlabel('Model Type')
                    plt.ylabel(f'{metric} Score')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_directory, f'{filename_base}_by_model_type_violinplot.png'))
                    plt.show()

            # Grouped Scatter Plot - e.g., ROUGE-1 F1 vs. BERT F1
            if 'ROUGE-1 F1' in df.columns and 'BERT F1' in df.columns:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x='ROUGE-1 F1', y='BERT F1', hue='model_type', data=df, s=50, alpha=0.7)
                plt.title('ROUGE-1 F1 vs. BERT F1 by Model Type')
                plt.xlabel('ROUGE-1 F1 Score')
                plt.ylabel('BERT F1 Score')
                plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left') # Adjust legend position
                plt.tight_layout()
                plt.savefig(os.path.join(output_directory, 'ROUGE-1_vs_BERT_by_model_type_scatter.png')) # Filename updated
                plt.show()
        # --- End Grouped Bivariate Plots ---


        # 4.2 Correlation Matrix Heatmap - Display the linear correlation between multiple variables (Existing, updated for ROUGE-L)
        correlation_cols_for_heatmap = ['ROUGE-1 F1', 'BERT F1', 'Original Chinese Character Count', 'Summary English Word Count', 'Summarization Time', 'Translation Time']
        if 'ROUGE-L F1' in df.columns:
            correlation_cols_for_heatmap.append('ROUGE-L F1')

        # Ensure all columns exist before calculating correlation
        existing_heatmap_cols = [col for col in correlation_cols_for_heatmap if col in df.columns]
        if existing_heatmap_cols:
            correlation_matrix = df[existing_heatmap_cols].corr()
            print("\n--- Correlation Matrix (Pearson) ---") # Specify Pearson correlation
            print(correlation_matrix)

            plt.figure(figsize=(12, 10))   # Increase figure size to accommodate labels
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # fmt=".2f" for number formatting
            plt.title('Correlation Matrix of Evaluation Metrics and Processing Time')

            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
            plt.yticks(rotation=0, va='top')      # Rotate y-axis labels (no rotation here, adjust vertical alignment)
            plt.tight_layout()                  # Adjust layout to prevent labels from being cut off

            plt.savefig(os.path.join(output_directory, 'correlation_matrix_heatmap.png'))
            plt.show()
        else:
            print("\nWarning: No relevant numerical columns found for Pearson correlation heatmap.")


        # Pair Plot / Scatter Matrix Plot
        print("\n--- Generating Pair Plot of Key Metrics ---")
        # Select numerical columns for the Pair Plot
        pairplot_cols = ['ROUGE-1 F1', 'BERT F1', 'Original Chinese Character Count', 'Summarization Time']
        if 'ROUGE-L F1' in df.columns:
            pairplot_cols.append('ROUGE-L F1')
        
        # Filter for only existing columns in the DataFrame
        existing_pairplot_cols = [col for col in pairplot_cols if col in df.columns]

        df_for_pairplot = df[existing_pairplot_cols].copy()

        # Optionally add a hue variable if 'model_type' exists for grouping
        if 'model_type' in df.columns:
            df_for_pairplot['model_type'] = df['model_type']
            # Drop rows with NaN in the hue column if it's used for grouping
            df_for_pairplot.dropna(subset=existing_pairplot_cols + ['model_type'], inplace=True)
            if not df_for_pairplot.empty:
                sns.pairplot(df_for_pairplot, hue='model_type', diag_kind='kde')
            else:
                print("Warning: DataFrame is empty after dropping NA for Pair Plot with 'model_type'. Skipping Pair Plot.")
        else:
            df_for_pairplot.dropna(inplace=True) # Drop rows with NaN for numerical columns
            if not df_for_pairplot.empty:
                sns.pairplot(df_for_pairplot, diag_kind='kde')
            else:
                print("Warning: DataFrame is empty after dropping NA for Pair Plot. Skipping Pair Plot.")

        if not df_for_pairplot.empty:
            plt.suptitle('Pair Plot of Key Metrics', y=1.02) # Adjust overall title position
            plt.savefig(os.path.join(output_directory, 'key_metrics_pairplot.png'))
            plt.show()


    except FileNotFoundError:
        print(f"Error: The file '{input_csv_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        # Provide a more detailed hint if it's a KeyError (often due to missing column)
        if isinstance(e, KeyError):
            print(f"Please check if the column '{e.args[0]}' exists in your CSV file.")