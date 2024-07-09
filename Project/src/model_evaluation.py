from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        ''' return -> f1,accuracy,precision,recall'''
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        return f1,accuracy,precision,recall
