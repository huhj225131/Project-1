import pickle
with open("parameters.pkl", "rb") as f:
    parameters = pickle.load(f)
print("Parameters đã được tải:")
print(parameters)