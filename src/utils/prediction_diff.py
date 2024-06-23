def prediction_add_diff(curr_predictions, prev_prediction):
    return [curr_predict for curr_predict in curr_predictions if curr_predict not in prev_prediction]
