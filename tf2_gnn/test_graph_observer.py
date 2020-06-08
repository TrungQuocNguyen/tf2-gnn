import pickle
from tf2_gnn import GraphObserverModel
path_train_file = '/home/trung/tf2_gnn_folder/tf2-gnn/train-data/normed_data/highway/lane_change_left/dataset_1591439765.pickle'
path_validation_file = '/home/trung/tf2_gnn_folder/tf2-gnn/train-data/normed_data/highway/lane_change_left/dataset_1591439768.pickle'
with open(path_train_file, 'rb') as f1:
    train_data = pickle.load(f1)
f1.close()
with open(path_validation_file, 'rb') as f2:
    validation_data = pickle.load(f2)
f2.close()


observer_model = GraphObserverModel(model = "RGCN", 
                                    task = "NodeLevelRegression",
                                    max_epochs= 200, 
                                    patience=30
                                    )


#predict output with an untrained model
(predicted_targets_random, true_targets_random) = observer_model(train_data)


#training model 
print("training model...")
observer_model.fit(train_data, train_data)
#print("Selected dataset:",path_train_file)
#print("The selected dataset is of type", train_data.__class__, "and consists of", len(train_data),"datapoints.")
#print("\tOne datapoint is of type", train_data[0].__class__, "with", train_data[0].keys())
print("-------------------------------------------------------------------------------------------------------")
print("PREDICTED OUTPUT BEFORE TRAIN")
print("predicted data")
print(predicted_targets_random)
print("true data")
print(true_targets_random)

print("-------------------------------------------------------------------------------------------------------")
print("PREDICTED OUTPUT AFTER TRAIN")
#predict output with trained model
(predicted_targets, true_targets) = observer_model(train_data)
print("predicted data")
print(predicted_targets)
print("true data")
print(true_targets)