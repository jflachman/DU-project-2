# Metrics

Note:  Also see [Regression and Classification Model Evaluation](https://arifromadhan19.medium.com/part-1-regression-and-classification-model-evaluation-bc7f6ab3b4dd#:~:text=R%2Dsquared%20%3D%20Explained%20variation%20%2F%20Total%20variation&text=In%20essence%2C%20R%2DSquared%20represents,always%20between%200%20and%20100%25.)

## Reference: Gemini AI

### Balanced Accuracy:

- This metric is particularly useful for evaluating models on multiclass classification tasks with imbalanced datasets.expand_more
- It addresses the issue where a model might achieve high overall accuracy by simply predicting the majority class most of the time.
- Balanced accuracy is essentially the average of recall (also called sensitivity) obtained for each class.expand_more
- Recall measures how well your model identifies positive examples (of a particular class) that actually are positive.expand_more
- By averaging recall across all classes, balanced accuracy provides a more balanced assessment of the model's performance, especially when dealing with unequal class distributions.

### ROC AUC Score (Area Under the Receiver Operating Characteristic Curve):

- This metric is used for binary classification problems.expand_more
- It evaluates a model's ability to distinguish between positive and negative classes.expand_more
- The ROC curve plots the True Positive Rate (TPR) (recall) against the False Positive Rate (FPR) for different classification thresholds.expand_more
- AUC (Area Under the Curve) essentially measures the total area underneath the ROC curve.expand_more
- An AUC score of 1 indicates perfect performance, while 0.5 represents a random guess.expand_more Scores closer to 1 are generally better.exclamation

### Mean Squared Error (MSE):

- This metric is commonly used for evaluating the performance of regression models.expand_more
- It measures the average squared difference between the predicted values by the model and the actual values.expand_more
- Lower MSE indicates a better fit, meaning the model's predictions are on average closer to the actual values.expand_more
- It's important to note that MSE can be sensitive to outliers, so consider other metrics like Root Mean Squared Error (RMSE) which is the square root of MSE for better interpretability.

## Reference: Definitions from Claude AI

### Accuracy:

Accuracy is the ratio of correctly predicted instances (both true positives and true negatives) to the total number of instances. It measures the overall correctness of the classifier. The formula is:

    codeAccuracy = (TP + TN) / (TP + TN + FP + FN)

Accuracy is a good measure when the classes are balanced, but it can be misleading when dealing with imbalanced datasets.

### Precision:

Precision is the ratio of true positives to the sum of true positives and false positives. It measures the ability of the classifier to avoid labeling negative instances as positive. The formula is:

    codePrecision = TP / (TP + FP)

Precision is important in scenarios where false positives are more costly than false negatives, such as spam detection or fraud detection.

### Recall (Sensitivity or True Positive Rate):

Recall is the ratio of true positives to the sum of true positives and false negatives. It measures the ability of the classifier to find all positive instances. The formula is:

Copy codeRecall = TP / (TP + FN)
Recall is important in scenarios where false negatives are more costly than false positives, such as disease diagnosis or intrusion detection.

### F1-score:

The F1-score is the harmonic mean of precision and recall. It provides a balanced measure that takes both metrics into account. The formula is:

    codeF1-score = 2 * ((Precision * Recall) / (Precision + Recall))

The F1-score is useful when you want to consider both precision and recall, and their relative importance is equal.

### Specificity (True Negative Rate):

Specificity is the ratio of true negatives to the sum of true negatives and false positives. It measures the ability of the classifier to correctly identify negative instances. The formula is:

    codeSpecificity = TN / (TN + FP)

Specificity is important in scenarios where false positives are more costly than false negatives, such as medical screening tests or spam detection.

### False Positive Rate:

The false positive rate is the ratio of false positives to the sum of false positives and true negatives. It measures the probability of incorrectly labeling a negative instance as positive. The formula is:

    codeFalse Positive Rate = FP / (FP + TN)

The false positive rate is useful in evaluating the trade-off between sensitivity and specificity, and it is often used in conjunction with the true positive rate (recall) in ROC (Receiver Operating Characteristic) curve analysis.

### Matthews Correlation Coefficient (MCC):

The Matthews Correlation Coefficient is a balanced measure that takes into account true and false positives and negatives. It returns a value between -1 and +1, where +1 represents a perfect prediction, 0 represents a random prediction, and -1 represents an inverse prediction. The formula is:

    codeMCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

The MCC is considered a reliable and robust metric, especially for imbalanced datasets, as it handles class imbalance well.
The choice of metric depends on the specific problem and the relative importance of false positives and false negatives. It's generally recommended to evaluate multiple metrics to get a comprehensive understanding of the model's performance, especially when dealing with imbalanced datasets or scenarios where the costs of different types of errors are not equal.