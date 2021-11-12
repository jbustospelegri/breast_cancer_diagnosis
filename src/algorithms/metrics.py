from keras import backend


def f1_score(y_true, y_pred):
    """
    Función utilizada para calcular la métrica F1 a partir de las predicciones y los labels verdaderos
    :param y_true: array con labels verdaderos
    :param y_pred: array con las predicciones
    :return: métrica f1
    """

    def get_recall(true_label, pred_label):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = backend.sum(backend.round(backend.clip(true_label * pred_label, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(true_label, 0, 1)))
        return true_positives / (possible_positives + backend.epsilon())

    def get_precision(true_label, pred_label):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = backend.sum(backend.round(backend.clip(true_label * pred_label, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(pred_label, 0, 1)))
        return true_positives / (predicted_positives + backend.epsilon())

    precision = get_precision(y_true, y_pred)

    recall = get_recall(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + backend.epsilon()))