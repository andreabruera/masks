import argparse
import mne
import collections
import numpy

### Finds and writes to file for each searchlight cluster 1. the number 2. the center 3+ the neighbouring ones at a max distance of 20mm

def get_searchlight_groups(max_distance=0.02):

    max_distance = float(max_distance)
    montage = mne.channels.make_standard_montage('biosemi128', head_size=0.088)
    positions = montage.get_positions()['ch_pos']
    position_indices = {electrode : electrode_index for electrode_index, electrode in enumerate(positions.keys())}

    ### Rescaling so that the minimum coordinate value is 0
    scaler = list()
    for i in range(3):
        scaler.append(abs(min([v[i] for k, v in positions.items()])))
    positions = {k : numpy.add(v, scaler) for k, v in positions.items()} 

    ### Now obtaining the searchlight group with tolerance 20mm
    searchlight_groups = collections.defaultdict(list)
    for channel, position in positions.items():
        neighbour_tolerance = [(k-max_distance, k+max_distance) for k in position]
        for other_channel, other_position in positions.items():
            if other_channel != channel:
                marker = [False, False, False]
                for axis_index, window in enumerate(neighbour_tolerance):
                    if other_position[axis_index] >= window[0] and other_position[axis_index] <= window[1]:
                        marker[axis_index] = True
                if False not in marker:
                    searchlight_groups[(channel, position_indices[channel])].append(position_indices[other_channel])

    return searchlight_groups

parser = argparse.ArgumentParser()
parser.add_argument('--max_distance', default=0.02, help='Max distance between an electrode and its neighbours')
args = parser.parse_args()

searchlight_groups = get_searchlight_groups(args.max_distance)

with open('searchlight_clusters_{}mm.txt'.format(int(float(args.max_distance)*1000)), 'w') as o:
    o.write('Central electrode (CE) code\tCE index\tNeighbors\n')
    for channel_info, other_channels_list in searchlight_groups.items():
        o.write('{}\t{}\t'.format(channel_info[0], channel_info[1]))
        for other_channel_index in other_channels_list:
            o.write('{}\t'.format(other_channel_index))
        o.write('\n')
