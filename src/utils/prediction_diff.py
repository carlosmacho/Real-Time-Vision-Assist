def prediction_add_diff(curr_predictions, prev_predictions):
    # Identify and return new predictions that are not present in the previous predictions
    return [prediction for prediction in curr_predictions if prediction not in prev_predictions]
