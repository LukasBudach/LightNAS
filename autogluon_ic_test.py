import autogluon as ag
from autogluon import ImageClassification as task


def get_dataset():
    dataset = task.Dataset(name='FashionMNIST')
    test_dataset = task.Dataset(name='FashionMNIST', train=False)
    return dataset, test_dataset


def main():
    train, test = get_dataset()
    classifier = task.fit(train, epochs=5, ngpus_per_trial=1, verbose=False)


if __name__ == '__main__':
    main()
