from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import graphviz
import load_data
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn import metrics

class IRIS_CLASSIFIER(object):
    def __init__(self, feature_names, feature_num, test_size, fold_num):
        self.decision_tree_classifier = tree.DecisionTreeClassifier()
        self.features = feature_names
        self.feature_num = feature_num
        self.test_size = test_size
        self.fold_num = fold_num

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

    def split_data(self):
        train_input, test_input, train_label, test_label = train_test_split(self.inputs, self.targets, test_size=self.test_size, random_state=0)
        print("Train data: {}".format(train_input.shape))
        print("Test data: {}".format(test_input.shape))
        return train_input, train_label, test_input, test_label

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

    def scoring(self, test_label, test_preds):
        print("accuracy = ", metrics.accuracy_score(test_label, test_preds))
        print("micro precision = ", metrics.precision_score(test_label, test_preds, average='micro'))
        print("macro precision = ", metrics.precision_score(test_label, test_preds, average='macro'))
        print("weighted precision = ", metrics.precision_score(test_label, test_preds, average='weighted'))
        print("micro recall = ", metrics.recall_score(test_label, test_preds, average='micro'))
        print("micro f1 = ", metrics.f1_score(test_label, test_preds, average='micro'))
        print("micro beta 0.5 = ", metrics.fbeta_score(test_label, test_preds, beta=0.5, average='micro'))
        print("micro beta 1 = ", metrics.fbeta_score(test_label, test_preds, beta=1, average='micro'))
        print("micro beta 2 = ", metrics.fbeta_score(test_label, test_preds, beta=2, average='micro'))
        return


def main():
    # features
    feature_num = 4
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    test_size = 0.8
    fold_num = 3

    # 1. init
    iris_tree = IRIS_CLASSIFIER(feature_names, feature_num, test_size, fold_num)
    tree_model = iris_tree.decision_tree_classifier

    # 2. manually load data
    data_path = "../data/iris.data"
    iris_tree.load_data(data_path)
    inputs = iris_tree.inputs
    targets = iris_tree.targets

    # samples
    iris_tree.get_samples()

    # 3. classify
    model = iris_tree.classifier(inputs, targets)
    # scores = model.score(inputs, targets)
    # print(scores)

    # training
    train_input, train_label, test_input, test_label = iris_tree.split_data()
    trained_model = iris_tree.classifier(train_input, train_label)
    
    # scoring
    scores = trained_model.score(test_input, test_label)
    print("test scores = ", scores)
    
    # predict
    test_preds = trained_model.predict(test_input)
    print(test_preds)
    print(test_label)

    # metrics
    iris_tree.scoring(test_label, test_preds)

    # cross validation
    # cross_scores = cross_val_score(tree_model, inputs, targets, cv=iris_tree.fold_num)
    cross_scores = cross_validate(tree_model, inputs, targets, cv=iris_tree.fold_num, scoring='precision_macro')
    print(cross_scores)

    # 4. plot
    # iris_tree.plotting_tree(outputs)
    iris_tree.export_graph(model)


if __name__ == '__main__':
    main()
