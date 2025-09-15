1. Data Preparation
The raw weekly data was preprocessed and engineered to prepare it for modeling.

Handling Seasonality and Trend:

Although time-based features (year, month, quarter) were created to capture seasonality, they were ultimately excluded from the final model to create a simpler, more parsimonious model focused on direct business levers.

A week_number feature was created to serve as a linear time trend, but was also excluded for the same reason. Trend effects are implicitly captured to some extent by features like followers_growth.

Handling Zero-Spend Periods:

Many marketing spend channels have weeks with zero spend. Applying a standard logarithm (np.log) would result in negative infinity.

To handle this, a log-transform of the form np.log1p(x) (equivalent to log(x+1)) was applied to all spend variables and the target variable (revenue). This transformation gracefully handles zeros while also compressing the range of the variables and normalizing their distributions.

Feature Scaling and Transformations:

Scaling: StandardScaler was used on average_price and followers_growth to center them around zero with a standard deviation of one. This is crucial for models that are sensitive to feature magnitudes, and while Random Forest is not, it's good practice and aids in interpretability of effects like Partial Dependence.

Categorical Features: The promotions variable was converted into a categorical type and then one-hot encoded using pd.get_dummies. This allows the model to treat each promotion type as a distinct binary feature.

Feature Engineering: New features were created, such as followers_growth and scaled versions of email/SMS sends (emails_send_k, sms_send_k), to better capture marketing momentum and simplify coefficients.

2. Modeling Approach
A Random Forest Regressor was chosen for both stages of the model.

Why Random Forest?

Non-linearity: It can capture complex, non-linear relationships between marketing spend and revenue without requiring manual transformations (e.g., adstock or saturation effects).

Interactions: It automatically handles interaction effects between features (e.g., the impact of a promotion might be higher when average price is also high).

Robustness: It is less sensitive to outliers compared to linear models.

Feature Importance: It provides a built-in mechanism to rank features by their predictive power, which is essential for deriving business insights.

Hyperparameter Choices & Regularization:

The hyperparameters were chosen to regularize the model and prevent overfitting.

n_estimators=300: A sufficient number of trees to ensure stable predictions.

max_depth=6: Limits the depth of each tree, preventing it from fitting noise in the training data.

min_samples_split=5 & min_samples_leaf=3: Requires a minimum number of data points to make a split or form a leaf, which smooths the model and reduces variance.

max_features='sqrt': At each split, only a random subset of features is considered, which de-correlates the trees and improves generalization.

Validation Plan:

A simple time-based 80/20 split was used for validation. The first 80% of the data was used for training and the final 20% for testing.

This approach is crucial for time-series data as it ensures the model is validated on "unseen" future data, simulating a real-world forecasting scenario.

3. Causal Framing
The model is designed to address a key causal question: what is the impact of social media spend on revenue, considering that it also influences Google spend?

Mediator Assumption:

We assume that Google spend acts as a mediator for other social media channels. The causal path is: Social Spend → Google Spend → Revenue.

Ignoring this path would lead to double-counting the effect of social spend and misattributing its impact.

Two-Stage Approach:

To handle this, we use a two-stage approach to "block" the back-door path.

Stage 1 models the path Social Spend → Google Spend. The output, google_spend_pred, represents the portion of Google spend that is causally explained by the spend on Facebook, TikTok, Instagram, and Snapchat.

Stage 2 then models revenue using google_spend_pred instead of the actual google_spend. This isolates the indirect effect of social channels that flows through Google. The direct effects of other levers (email, price, etc.) are modeled alongside it.

Leakage and Confounding:

This structure helps prevent leakage from the mediating variable (Google spend). If we had included both facebook_spend and google_spend in a single model, the model would struggle to separate their correlated effects.

A potential unaddressed back-door path could be an unobserved confounding variable, such as a major competitor's campaign, which could simultaneously cause our brand to increase social/Google spend and also lead to lower revenue.

4. Diagnostics
Out-of-Sample Performance:

Stage 1 (Predicting Google Spend):

Test RMSE: 0.70

Test R²: 0.82

This indicates a strong fit. The model can explain 82% of the variance in Google spend based on other social media spends.

Stage 2 (Predicting Revenue):

Test RMSE: 0.12

Test R²: 0.74

The final revenue model explains 74% of the variance in out-of-sample weekly revenue, which is a robust result.

Stability Checks:

The current model uses a single train-test split. For a more rigorous assessment of stability, a rolling cross-validation or blocked cross-validation approach would be a valuable next step. This would involve training and testing the model on multiple sequential windows of data to ensure its performance is consistent over time.

Residual Analysis (Next Step):

The script does not currently include residual analysis. A recommended next step is to plot the residuals (predicted - actual) over time. This would help diagnose if the model has any systematic biases, such as failing to capture seasonality or a changing trend.

Sensitivity to Price and Promotions:

The Partial Dependence Plots (PDPs) provide excellent insight into model sensitivity.

The PDP for average_price_scaled shows a clear, strong positive relationship: as the scaled average price increases, so does log-revenue. This suggests the model has learned a strong price elasticity effect.

The PDPs for promotions would show the average change in revenue when a specific promotion is active, holding all other features constant.

5. Insights & Recommendations
Interpretation of Revenue Drivers:

Based on the Stage 2 feature importances, the primary drivers of revenue are:

Average Price (average_price_scaled): This is the single most important feature. Pricing strategy has a direct and powerful impact on weekly revenue.

Emails Sent (emails_send_k): Email marketing is a highly effective channel.

Mediated Google Spend (google_spend_pred): The portion of Google spend driven by social media activity is a significant contributor to revenue. This validates the hypothesis that social channels create demand that is later captured by search.

SMS Sent (sms_send_k): SMS is another effective direct marketing channel.

Promotions: Specific promotions, such as "Promotion 2," also show notable importance, indicating their effectiveness in driving short-term revenue lifts.

Risks and Considerations:

Collinearity: While the two-stage approach mitigates the collinearity between Google and social spend, other features might still be correlated. For example, emails_send_k and sms_send_k may be highly correlated if campaigns are often launched simultaneously. This can make it difficult to isolate their individual impacts perfectly.

Mediated Effects: The key insight is that the value of social media is not just its direct impact but also its indirect impact via search. A recommendation would be to coordinate social and search marketing strategies. When planning a large social campaign, the search team should be prepared for an increase in branded search volume. Cutting social media spend may lead to an unforeseen drop in search-driven revenue.

Model Limitations: A Random Forest model provides feature importance but not simple, linear coefficients like a regression model. The PDPs are essential for understanding the direction and magnitude of each feature's effect.
