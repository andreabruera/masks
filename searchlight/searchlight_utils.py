import scipy
import collections

from scipy import stats
from tqdm import tqdm


class SearchlightClusters:

    def __init__(self, max_distance=20):

        self.max_distance = max_distance
        self.index_to_code = self.indices_to_codes()
        self.neighbors = self.read_searchlight_clusters()
        self.mne_adjacency_matrix = self.create_adjacency_matrix()

    def create_adjacency_matrix(self):
        data = list()
        indices = list()
        index_pointer = [0]
        for i, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]
            for neighbor in v:
                indices.append(int(neighbor))
                data.append(1)
            index_pointer.append(len(indices))

        ### Just checking everything went fine
        mne_sparse_adj_matrix = scipy.sparse.csr_matrix((data, indices, index_pointer), dtype=int)
        for ikv, kv in enumerate(self.neighbors.items()):
            v = kv[1][1:]

            assert [i for i, k in enumerate(mne_sparse_adj_matrix.toarray()[ikv]) if k == 1] == v

        return mne_sparse_adj_matrix 

    def indices_to_codes(self):

        index_to_code = collections.defaultdict(str)
        with open('searchlight/searchlight_clusters_{}mm.txt'.format(self.max_distance), 'r') as searchlight_file:
            for l in searchlight_file:
                if 'CE' not in l:
                    l = l.strip().split('\t')
                    index_to_code[int(l[1])] = l[0]

        return index_to_code

    def read_searchlight_clusters(self):

        searchlight_clusters = collections.defaultdict(list)

        with open('searchlight/searchlight_clusters_{}mm.txt'.format(self.max_distance), 'r') as searchlight_file:
            for l in searchlight_file:
                if 'CE' not in l:
                    l = [int(i) for i in l.strip().split('\t')[1:]]
                    searchlight_clusters[l[0]] = l

        return searchlight_clusters

def run_searchlight(evoked_dict, word_combs, computational_scores, time_points, temporal_window_size): 

    ### Loading the searchlight clusters
    searchlight_clusters = SearchlightClusters()

    current_condition_rho = collections.defaultdict(list)

    for center in tqdm(range(128)):

        relevant_electrode_indices = searchlight_clusters.neighbors[center]

        for t in time_points:

            eeg_similarities = list()

            relevant_time_indices = [t+i for i in range(temporal_window_size)]

            for word_one, word_two in word_combs:

                eeg_one = list()
                eeg_two = list()

                for relevant_time in relevant_time_indices:
                    for relevant_electrode in relevant_electrode_indices:
                    
                        eeg_one.append(evoked_dict[word_one][relevant_electrode, relevant_time])
                        eeg_two.append(evoked_dict[word_two][relevant_electrode, relevant_time])

                word_comb_score = scipy.stats.spearmanr(eeg_one, eeg_two)[0]
                eeg_similarities.append(word_comb_score)

            rho_score = scipy.stats.spearmanr(eeg_similarities, computational_scores)[0]
            current_condition_rho[center].append(rho_score)

    return current_condition_rho
