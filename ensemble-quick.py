import pandas as pd

# Step 1: Normalize the Weights
weights = {
    'model1': 0.77534 / 0.78528,  # Normalized Weight 1
    'model2': 1.0,                # Normalized Weight 2
    'model3': 0.68986 / 0.78528   # Normalized Weight 3
}

# Print normalized weights for verification
print("Normalized Weights:")
for model, weight in weights.items():
    print(f"{model}: {weight:.5f}")

# Step 2: Read the Prediction Files
pred1 = pd.read_csv('test_predictions2_resnet18_Entire.csv')
pred2 = pd.read_csv('test_predictions2_efficientnet_b0_Entire.csv')
pred3 = pd.read_csv('test_predictions_pretrained.csv')

# Ensure that the predictions are aligned by 'id'
predictions = pred1[['ID']].copy()
predictions['model1'] = pred1['Class']
predictions['model2'] = pred2['Class']
predictions['model3'] = pred3['Class']

# Step 3 & 4: Weighted Voting and Determine Final Predictions
def weighted_vote(row):
    from collections import defaultdict
    vote_counts = defaultdict(float)
    
    # Collect predictions and their corresponding weights
    preds = {
        row['model1']: weights['model1'],
        row['model2']: weights['model2'],
        row['model3']: weights['model3']
    }
    
    # Sum the weights for each class label
    for label, weight in preds.items():
        vote_counts[label] += weight
    
    # Find the class with the highest total weight
    max_weight = max(vote_counts.values())
    winning_classes = [label for label, weight in vote_counts.items() if weight == max_weight]
    
    # Handle ties
    if len(winning_classes) == 1:
        return winning_classes[0]
    else:
        # In case of a tie, select the class from the highest-weighted model
        for model in ['model2', 'model1', 'model3']:
            if row[model] in winning_classes:
                return row[model]

predictions['final_prediction'] = predictions.apply(weighted_vote, axis=1)
predictions['Class']=predictions['final_prediction']
# Step 5: Save the Final Predictions
predictions[['ID', 'Class']].to_csv('ensemble-quick.csv', index=False)

print("Weighted majority voting completed. Final predictions saved to 'final_predictions.csv'.")
