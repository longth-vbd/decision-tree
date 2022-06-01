from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import graphviz
import load_data


class IRIS_CLASSIFIER(object):
    def __init__(self, feature_names, feature_num):
        self.decision_tree_classifier = tree.DecisionTreeClassifier()
        self.features = feature_names
        self.feature_num = feature_num

    def load_data(self, data_path):
        iris = load_data.manually_load_data(data_path, self.feature_num)
        self.inputs = iris["data"]
        self.targets = iris["target"]
        self.labels = iris["target_names"]
        self.input_size = len(self.inputs)
        self.target_size = len(self.targets)

        print("Inputs: size={}".format(self.inputs.shape))
        print("targets: size={}".format(self.targets.shape))
        print("\nFeature names: {}".format(self.features))
        print("Target names: {}\n".format(self.labels))

    def get_samples(self):
        input_samples = self.inputs[:5]
        target_samples = self.targets[:5]

        print("{}\t\t{}\t\t\t\t{}".format("id", "inputs", "label"))
        for sid, (input, target) in enumerate(zip(input_samples, target_samples)):
            print("{}\t{}\t\t{}".format(sid, input, target))

        # assert len(self.features) != len(input_samples[0])
        # assert len(self.labels) != len(target_samples[0])

        print()
        for sid, (input, target) in enumerate(zip(input_samples, target_samples)):
            features_info = {}
            for id, feature in enumerate(self.features):
                value = input[id]
                features_info[feature] = value
            label_info = self.labels[target]
            print("{}, {}: {}".format(sid, features_info, label_info))

    def classifier(self, inputs, targets):
        classified_tree = self.decision_tree_classifier.fit(inputs, targets)
        return classified_tree

    def plotting_tree(self, classified_tree):
        plt.figure()
        plot_tree(classified_tree, filled=True)
        plt.title("Decision tree trained on all the iris features")
        plt.show()

    def export_graph(self, classified_tree):
        dot_data = tree.export_graphviz(classified_tree, out_file=None, feature_names=self.features, class_names=self.labels, filled=True, rounded=True, special_characters=True)
        graph = graphviz.Source(dot_data)
        # print(graph)
        graph.render("iris")


def main():
    # features
    feature_num = 4
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # 1. init
    iris_tree = IRIS_CLASSIFIER(feature_names, feature_num)
    model = iris_tree.decision_tree_classifier

    # 2. manually load data
    data_path = "../data/iris.data"
    iris_tree.load_data(data_path)
    inputs = iris_tree.inputs
    targets = iris_tree.targets

    # samples
    iris_tree.get_samples()

    # 3. classify
    outputs = iris_tree.classifier(inputs, targets)

    # 4. plot
    # iris_tree.plotting_tree(outputs)
    iris_tree.export_graph(outputs)


if __name__ == '__main__':
    main()
