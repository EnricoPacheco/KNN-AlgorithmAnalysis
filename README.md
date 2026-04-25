# Machine Learning I (CC2008)

This is the repository for the first assignment of the 'Machine Learning I' course, which focuses on implementing changes in a machine learning algorithm.

Group: [Enrico Pacheco Sanchez](https://github.com/EnricoPacheco) | [Enzo Nascentes Grigório](https://github.com/egrigorio) |  Matheus Goncalves Guerra

# Phase 1
  In the initial phase of this project, we established a robust experimental baseline by using a manual k-Nearest Neighbors (kNN) classifier. Our focus was on analyzing the model's behavior when exposed to Group 1: Noise and Outliers. To ensure a deep understanding of the algorithm's mechanics, we avoided high-level library implementations and developed the logic for Euclidean distance and majority voting using as base the implementation in: https://github.com/rushter/MLAlgorithms/tree/master .
  
## Data Preprocessing Pipeline
  Before running the benchmark, we developed an automated pipeline to handle diverse datasets:
   - Label Encoding: Categorical features and target labels were converted into numerical formats to allow for distance calculations.
   - Missing Values: Gaps in the data were handled through mean imputation to maintain dataset integrity.
   - Min-Max Scaling: All features were normalized to a $[0, 1]$ range. This is critical for kNN as it prevents features with larger numerical scales from disproportionately influencing the distance metrics.

## Experimental Protocol
  We conducted an extensive empirical study across approximately 50 noisy datasets:
   - 10-fold Cross-Validation: To ensure statistical stability, every dataset was split into 10 folds, training on 9 and testing on 1, repeating the process 10 times.
   - Hyperparameter Tuning: We evaluated $k$ values of $1, 3, 5,$ and $11$ to observe the trade-off between local sensitivity and global smoothing.
   - Performance Metrics: Results were recorded as Mean Accuracy accompanied by the Standard Error of the Mean (SEM) to measure the consistency of the model across different folds.

## Statistical Validation
  To prove the scientific validity of our findings, we applied the Friedman Test. This non-parametric test confirmed that the differences in performance between the chosen $k$ values were statistically significant ($p < 0.05$), rejecting the null hypothesis that all configurations performed equally. Our results indicated that while $k=1$ is highly susceptible to noise, higher values of $k$ act as a natural filter, smoothing decision boundaries and improving robustness. This baseline now serves as the foundation for Phase 2, where we introduce Local Outlier Factor (LOF) to further optimize performance by proactively cleaning the training data.
