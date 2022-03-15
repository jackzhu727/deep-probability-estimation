import pickle
from torch.utils.data import Dataset


class SurvivalDataset(Dataset):
    def __init__(self, data_path):
        self.embedding_path = "{}_embeddings.pkl".format(data_path)
        self.outcome_path = "{}_outcomes.pkl".format(data_path)
        self.embeddings = pickle.load(open(self.embedding_path, 'rb'))
        self.outcome = pickle.load(open(self.outcome_path, 'rb'))
        self.slide_ids = list(self.outcome.keys())
        self.proposed_probs = np.full(len(self.outcome), 0.5)

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, index):
        slide_id = self.slide_ids[index]
        outcomes = self.outcome[slide_id]
        label = outcomes[0]
        gt_prob = outcomes[1]
        prob = self.proposed_probs[index]
        return self.embeddings[slide_id], label, prob, int(slide_id), gt_prob