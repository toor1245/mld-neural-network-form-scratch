import csv
import os.path

from os import getcwd
from src.datasets.mnist_sample import MnistSample


class Dataset:

    @staticmethod
    def _load(path: str):
        res = []

        mnist_path = os.path.join(getcwd(), "src", "datasets", path)
        with open(mnist_path, 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader)

            for data in csvreader:
                label = data[0]

                pixels = data[1:]

                columns = [float(x) for x in pixels]

                sample = MnistSample(label, columns)
                res.append(sample)

        return res

    @staticmethod
    def load() -> tuple[list[MnistSample], list[MnistSample]]:
        train = Dataset._load("mnist_train.csv")
        test = Dataset._load("mnist_test.csv")
        return train, test
