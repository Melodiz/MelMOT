import json

def read_clusters(clusters_file):
    with open(clusters_file, 'r') as f:
        data = json.load(f)
    return data['clusters']

def read_original_ids(tracklet_file):
    with open(tracklet_file, 'r') as f:
        data = json.load(f)
    return list(data['tracklets'].keys())

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def create_remap(tracklets_file, clusters_file):
    clusters = read_clusters(clusters_file)
    original_ids = list(map(int, read_original_ids(tracklets_file)))
    remap = {}
    for original_id in original_ids:
        for i in range(len(clusters)):
            if original_id in clusters[i]:
                remap[original_id] = i+1
                break
    return remap

if __name__ == "__main__":
    tracklets_file = 'results/simple_1_tracklets.json'
    clusters_file = 'simple_clusters.json'
    remap = create_remap(tracklets_file, clusters_file)
    print(remap)