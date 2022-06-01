import numpy as np

def manually_load_data(data_path, feature_num):

    inputs = []
    targets = []

    with open(data_path) as fi:
        for line in fi:
            values = line.strip().split(",")
            if len(values) > feature_num:

                input = values[:feature_num]
                input = [float(st) for st in input]
                inputs.append(input)

                target = values[-1]
                target = target.replace("Iris-", "")
                targets.append(target)

    # assert len(inputs) != len(labels)
    print("Loaded data from {}".format(data_path))
    target_names = list(sorted(set(targets)))
    labels = {val:id for id, val in enumerate(target_names)}
    targets = [labels[val] for val in targets]

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    print("inputs: size={}".format(inputs.shape))
    print("targets: size={}".format(targets.shape))
    print("labels: {}".format(labels))

    return {"data": inputs, "target": targets, "features": labels, "target_names": target_names}

if __name__ == '__main__':
    data_path = "../data/iris.data"
    feature_num = 4
    manually_load_data(data_path, feature_num)