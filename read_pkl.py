import pickle

with open('/Users/apoorvgarg/PycharmProjects/BTP-Slice-RL/saved_weight.pickle', 'rb') as f:
    content = pickle.load(f)
    print(content)