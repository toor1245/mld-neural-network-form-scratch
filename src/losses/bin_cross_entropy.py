from src.core.matrix import Matrix as mt


class BinaryCrossEntropy:
    @staticmethod
    def bce(y_true: mt, y_pred: mt):
        left_op = mt.dot(-y_true, mt.log(y_pred))
        right_op = mt.dot(mt.sub_num(1.0, y_true), mt.log(mt.sub_num(1.0, y_pred)))

        return (left_op - right_op).mean()

    @staticmethod
    def bce_prime(y_true: mt, y_pred: mt):
        diff = mt.sub(mt.div_mt(mt.sub_num(1.0, y_true), mt.sub_num(1.0, y_pred)), mt.div_mt(y_true, y_pred))
        return mt.div(diff, y_true.length)
