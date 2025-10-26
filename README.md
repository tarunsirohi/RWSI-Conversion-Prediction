# Conversion Prediction Project

## Overview

This project explores and compares multiple machine learning models to predict user conversion (binary classification) in a dataset with class imbalance. The analysis covers every step from data preparation to model selection, hyperparameter tuning, evaluation, feature importance analysis, and final recommendations.

## Table of Contents

1. [Dataset Description](#dataset-description)
    - [Feature Definitions](#feature-definitions)
2. [Objective](#objective)
3. [Data Preparation and Exploration](#data-preparation-and-exploration)
4. [Model Selection and Rationale](#model-selection-and-rationale)
5. [Modeling Workflow](#modeling-workflow)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results and Insights](#results-and-insights)
8. [Feature Importance Analysis](#feature-importance-analysis)
9. [Final Conclusions and Recommendations](#final-conclusions-and-recommendations)
10. [Appendix: Code Explanations](#appendix-code-explanations)

## Dataset Description

The analysis uses the Retail Web Session Intelligence (RWSI) dataset, which simulates customer journeys on an e-commerce platform specializing in consumer and lifestyle products. Each row represents an anonymized user session, capturing a mix of behavioral, engagement, contextual, and outcome variables.

### Feature Definitions

| Column Name             | Description                                                                                                                                      |
|------------------------ |--------------------------------------------------------------------------------------------------------------------------------------------------|
| **SessionID**           | Unique alphanumeric identifier for each session (generated ID).                                                                                  |
| **AdClicks**            | Number of ad banners clicked during the session (0–4). Acts as a soft proxy for ad engagement.                                                   |
| **InfoSectionCount**    | Number of times a user accessed informational or support sections (e.g., FAQs, company info) during the session.                                 |
| **InfoSectionTime**     | Total time (in seconds) spent in informational/help sections. Indicates how much the user relied on non-product content before making decisions. |
| **HelpPageVisits**      | Count of dedicated help or guidance pages visited (e.g., "how to use", "warranty info").                                                         |
| **HelpPageTime**        | Cumulative time spent on help pages. Longer durations might suggest confusion or detailed exploration.                                            |
| **ItemBrowseCount**     | Number of product pages viewed in the session. A strong proxy for product discovery activity.                                                    |
| **ItemBrowseTime**      | Total time spent on product-related pages. Correlates with browsing depth or comparison behavior.                                                |
| **ExitRateFirstPage**   | Ratio of sessions that ended after the first page view. Measures immediate disengagement or bounce likelihood.                                   |
| **SessionExitRatio**    | Overall exit probability based on the number of pages viewed vs. total exits.                                                                    |
| **PageEngagementScore** | Derived score indicating how valuable or interactive the pages were (weighted sum of dwell times and interactions).                              |
| **HolidayProximityIndex** | Index (0–1) representing how close the session date is to major holidays or campaigns; higher means closer to a key retail period.            |
| **VisitMonth**          | Encoded month of visit (1–12). Useful for detecting seasonality or monthly behavior trends.                                                      |
| **UserPlatformID**      | Encoded identifier for the user's operating platform (Windows, Mac, iOS, Android, etc.).                                                        |
| **WebClientCode**       | Encoded browser identifier (e.g., Chrome, Edge, Safari).                                                                                        |
| **MarketZone**          | Encoded global region or market area (e.g., North America, Europe, Asia-Pacific, etc.).                                                         |
| **TrafficSourceCode**   | Encoded numeric tag for inbound traffic type (e.g., Organic, Paid Ads, Referral, Direct).                                                       |
| **UserCategory**        | Encoded user classification (e.g., New, Returning, or Loyal). Reflects behavioral grouping rather than identity.                                 |
| **IsWeekendVisit**      | Boolean indicator (0/1) showing if the session occurred on a weekend.                                                                           |
| **MonetaryConversion**  | (Target Variable) Binary variable (1 = the session resulted in a transaction or conversion, 0 = no conversion).                                 |

### Data Challenges

- Contains missing values and correlated features to simulate realistic e-commerce analytics scenarios.
- Feature diversity encourages exploration of how device, behavior, and time-based factors relate to conversion.
- Designed for end-to-end predictive modeling, feature engineering, and customer analytics.

## Objective

The primary goal is to predict, as accurately as possible, which user sessions are most likely to result in a monetary conversion, using session-level behavioral and contextual data. The project also aims to interpret what drives successful outcomes, providing actionable insights for digital marketing and product teams.

## Data Preparation and Exploration

- The data is split into training and testing subsets to enable robust model evaluation.
- Exploratory data analysis (EDA) examines trends and patterns, such as differences in engagement metrics or session characteristics between converting and non-converting sessions.
- Special attention is given to handling missing values and correlated variables, as these are common in real-world datasets.
- Feature engineering is considered to extract additional signals where appropriate.

## Model Selection and Rationale

Several algorithms are evaluated for their suitability to the problem:

- **Logistic Regression:**  
  Chosen for its simplicity and interpretability, providing a strong baseline and clear insights into feature influence.

- **Random Forest:**  
  Selected for its ability to capture non-linear patterns, robustness to overfitting, and natural handling of feature importance. Particularly useful for imbalanced data through flexible class weighting.

- **XGBoost:**  
  Used for its state-of-the-art performance with tabular data, sophistication in handling class imbalance, and strong capabilities in capturing complex interactions.

Each model is assessed both with default parameters and after hyperparameter tuning, with a focus on optimizing recall for the conversion class.

## Modeling Workflow

1. Baseline Training:  
   Each model is first trained with default settings to establish a baseline for comparison.

2. Hyperparameter Tuning:  
   Grid search cross-validation is used to systematically explore parameter combinations (e.g., tree depth, learning rate, class weights) to find the best performing configuration. The F1-score is the primary tuning metric, balancing precision and recall.

3. Model Evaluation:  
   Trained models are evaluated on the test set using a range of metrics, with a focus on how well each model identifies actual converters.

4. Model Comparison:  
   Metrics for all models—accuracy, ROC AUC, precision, recall, F1-score, and confusion matrices—are compared side by side to identify strengths and weaknesses.

5. Feature Importance Analysis:  
   For the top-performing model, feature importances are extracted and visualized, highlighting which factors most influence conversion predictions.

## Evaluation Metrics

- Accuracy: Overall rate of correct predictions.
- ROC AUC: Measures the model’s ability to distinguish between classes across all thresholds.
- Precision (for conversions): Proportion of predicted conversions that were correct.
- Recall (for conversions): Proportion of actual conversions that were identified.
- F1-score: Harmonic mean of precision and recall, particularly important for imbalanced data.
- Confusion Matrix: Detailed breakdown of true/false positives/negatives for both classes.

Emphasis is placed on recall for the conversion outcome, as missing potential converters is typically costlier than a few false alarms.

## Results and Insights

### Logistic Regression (Tuned)
- Achieves balanced precision (0.54) and recall (0.67) for conversions.
- Provides a robust, interpretable baseline.

### Random Forest (Default)
- High precision (0.63) but low recall (0.48) for conversions; misses many actual converters.

### Random Forest (Tuned)
- Recall for conversions improves significantly (0.73), with a small trade-off in precision (0.57).
- Tuning class weights and tree parameters enables the model to identify more true converters.

### XGBoost (Default)
- Performance similar to default Random Forest: high precision, moderate recall.

### XGBoost (Tuned)
- Delivers the highest recall for conversions (0.75) with balanced precision (0.56), making it the most effective at finding actual converters.
- This model offers the best trade-off between false positives and missed conversions, aligning with the project’s goals.

## Feature Importance Analysis

After identifying the best model (tuned XGBoost), feature importance scores are extracted to determine which variables most strongly influence conversion predictions. The top features typically reflect user engagement (e.g., product page views, time spent browsing), session timing (e.g., proximity to holidays), and contextual factors (e.g., market region, traffic source).

A bar chart visualization is used to present the top 10 most impactful features, guiding business teams on which metrics most affect conversion outcomes.

## Final Conclusions and Recommendations

- Ensemble models, especially tuned XGBoost, outperform logistic regression in identifying user sessions likely to convert.
- Hyperparameter tuning and class imbalance strategies are critical for boosting recall and overall effectiveness.
- Feature importance insights can inform marketing strategies, UI/UX improvements, and targeted interventions.
- For real-world deployment, model monitoring, periodic retraining, and further feature engineering are recommended to sustain high performance.

## Appendix: Code Explanations

### Feature Importances Extraction

- The `.feature_importances_` attribute of tree-based models (Random Forest, XGBoost) provides a numeric score for each feature, reflecting its impact on model predictions.
- These scores are paired with feature names and visualized to highlight key drivers of conversion.

### Understanding the Confusion Matrix

- True Positives (TP): Correctly predicted conversions.
- False Positives (FP): Incorrectly predicted conversions.
- False Negatives (FN): Missed actual conversions.
- True Negatives (TN): Correctly predicted non-conversions.
- Higher recall indicates the model is effective at finding actual converters, while higher precision means fewer false alarms.

### Hyperparameter Tuning Rationale

- GridSearchCV is used to systematically test parameter combinations, optimizing for the F1-score to best handle imbalanced classes.

This documentation is designed to serve as a comprehensive reference, capturing all key steps, decisions, insights, and rationale throughout the project, from understanding retail session data to delivering actionable machine learning solutions.
