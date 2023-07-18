import pickle
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
import utils
from datetime import datetime
import submitit
import pandas as pd

# Omit walks from zero degree nodes 
# Document which nodes do not complete the walk
# Ignore nodes without a race label 

def get_W(G, node_i, attr_name, attr_values):
    """
    Random walk starting at node_i over the graph G. Each time a new attribute value is seen, the time step is recorded. Returns a list of time steps. The number of time steps may not equal the total number of attribute values because of the maximum number of steps. List of timesteps is empty if attribute for node_i is "NA" and walk never reaches non-NA nodes.
    """
    t = 0
    max_steps = 15 * len(G)
    seen_values = set()
    curr_node = node_i
    output = []
    while (len(seen_values) < len(attr_values)) and (t < max_steps):
        current_label = G.nodes[curr_node][attr_name]
        if (current_label not in seen_values) and (current_label in attr_values):
            seen_values.add(current_label)
            output.append({
                "t": t,
                "c": len(seen_values) / len(attr_values),
                "value": current_label 
            })

        neighbors = list(G.neighbors(curr_node))
        weights = np.array([G[curr_node][node_j]["weight"] for node_j in neighbors])
        next_node = np.random.choice(neighbors,
                                    p = weights / np.sum(weights))
        curr_node = next_node
        t += 1
    
    if len(output) == 0:
        print(node_i)
    return t, output

def get_W_bar(G, node_i, attr_name, attr_values, R):
    """
    Aggregates the time steps after running R random walks. Returns a union of all time steps, for a given step the average value of c, the fraction of labels seen, is reported. If all trials are empty, the walks only encounter "NA", an empty list is returned.
    """
    outputs = [get_W(G, node_i, attr_name, attr_values) for _ in range(R)]
    filtered_outputs = []
    for t, output in outputs:
        if len(output) > 0:
            filtered_outputs.append((t, output))
    if len(filtered_outputs) == 0:
        return []
    outputs = filtered_outputs #Note, the number of trials is < R after filtering
    
    max_t = np.max([output[1][-1]["t"] for output in outputs])
    Ws = [output[1] for output in outputs]
    average_c = []
    current_c = [0] * len(Ws)
    current_idx = [0] * len(Ws)
    
    ts = []
    for output in outputs:
        ts.extend([step["t"] for step in output[1]])
    for t in sorted(ts):
#     for t in range(max_t + 1): #can be optimized to not search all t
        updated = False
        for r in range(len(Ws)):
            if current_idx[r] >= len(Ws[r]): #done processing this walk
                continue
            W_r = Ws[r][current_idx[r]]
            if t == W_r["t"]:
                current_c[r] = W_r["c"]
                current_idx[r] += 1
                updated = True
        if updated:
            average_c.append({
                "t": t,
                "avg_c": np.mean(current_c)
            })            
    assert average_c[-1]["t"] == max_t
    
    return average_c

def get_C(W_bar, max_time, step_size = 0.001):
    """
    For a given value of c, calculates the minimum value of t such that the average c is >= c. Discretizes the space of c to [0, step_size, 2*step_size, ... , 1]. If a level of c is never reached, the value is set to max_time.
    """
    n = int((1 / step_size)) + 1
    C = np.zeros(n) #[0, step_size, 2*step_size, ... , 1]
    current_index = len(W_bar)
    for step in range(n):
        c = 1 - step * step_size
        if current_index > 0 and W_bar[current_index - 1]["avg_c"] >= c:
            current_index -= 1

        if current_index == len(W_bar):
            C[n - 1 - step] = max_time
        else:
            C[n - 1 - step] = W_bar[current_index]["t"]
    return C

def get_null_graph(G, attr_name):
    attr_counts = {}
    for node in G:
        if G.nodes[node][attr_name] not in attr_counts:
            attr_counts[G.nodes[node][attr_name]] = 0
        attr_counts[G.nodes[node][attr_name]] += 1
    
    G_null = G.copy()
    keys = [key for key in attr_counts]
    p = [attr_counts[key] / len(G) for key in attr_counts]
    attr_values = {node: np.random.choice(keys, p=p) for node in G_null}
    nx.set_node_attributes(G_null, attr_values, attr_name)
    return G_null

#metrics
def get_mu(C_s, C_s_null, step_size):
    n = int(1 / step_size) + 1
    mu = np.array([np.mean([C[i] for C in C_s]) for i in range(n)])
    mu_null = np.array([np.mean([C[i] for C in C_s_null]) for i in range(n)])
    return step_size * np.sum(np.abs(mu - mu_null))

def get_sigma(C_s, C_s_null, step_size):
    n = int(1 / step_size) + 1
    mu = np.array([max(np.mean([C[i] for C in C_s]), 0.001) for i in range(n)])
    mu_null = np.array([max(np.mean([C[i] for C in C_s_null]), 0.001) for i in range(n)])
    
    std = np.array([np.std([C[i] for C in C_s]) for i in range(n)])
    std_null = np.array([np.std([C[i] for C in C_s_null]) for i in range(n)])
    return step_size * np.sum(np.abs((std / mu) - (std_null / mu_null)))

# parallel utils
def metrics_per_year(pre_processed_graphs, year, attr_name, attr_values):
    output_year = {}
    for month in range(1, 13):
        if year == 2022 and month > 10:
            continue
#             fig, ax = plt.subplots()
        C_s = []
        C_s_null = []
        incomplete = []
        step_size = 0.001
        G = pre_processed_graphs[year][month]["G"]
        G_null = pre_processed_graphs[year][month]["G_null"]
        
        for node in tqdm(list(G.nodes)):
            W_bar = get_W_bar(G, node, attr_name, attr_values, 10)
            if len(W_bar) == 0: #remove when there are no more "NA" tracts
                continue
            if W_bar[-1]["avg_c"] < 1.0: 
                incomplete.append(node)
            W_bar_null = get_W_bar(G_null, node, attr_name, attr_values, 10)

            C_s.append(get_C(W_bar, 15 * len(G), step_size))
            C_s_null.append(get_C(W_bar_null, 15 * len(G_null), step_size))

#                 ax.plot([step["t"] for step in W_bar], [step["avg_c"] for step in W_bar], c="#1b9e77")
#                 ax.plot([step["t"] for step in W_bar_null], [step["avg_c"] for step in W_bar_null], c="black")

#             plt.savefig("temp/W_bar.pdf", bbox_inches="tight")

#             fig, ax = plt.subplots()
#             for C in C_s:
#                 ax.plot(np.arange(0, 1 + step_size, step_size), C, c="#1b9e77")
#             for C in C_s_null:
#                 ax.plot(np.arange(0, 1 + step_size, step_size), C, c="black")
#             plt.savefig("temp/C.pdf", bbox_inches="tight")


        output_year[month] = {
            "Date": datetime(year, month, 1),
            "C_s": C_s,
            "C_s_null": C_s_null,
            "Mu": get_mu(C_s, C_s_null, step_size),
            "Sigma": get_sigma(C_s, C_s_null, step_size),
            "Incomplete Nodes": incomplete
        }
    return output_year
        
if __name__ == "__main__":
#     with open("temp/labeled_network.pickle", "rb") as pickleFile:
#         G = pickle.load(pickleFile)
#         G = G.subgraph(max(nx.weakly_connected_components(G), key = len))
#         G_null = get_null_graph(G, "Race")
    races = ["Hispanic or Latino", "White", "Black", "Asian"]

    with open("intermediates/station_name_to_id.pickle", "rb") as pickleFile:
        station_name_to_id = pickle.load(pickleFile)
    with open("intermediates/station_census_df.pickle", "rb") as pickleFile:
        station_census_df = pickle.load(pickleFile)
    with open("intermediates/station_networks_by_month.pickle", "rb") as pickleFile:
        station_networks = pickle.load(pickleFile)

    # preprocess 
    output = {}
    pre_processed_graphs = {}
    
    for year in range(2015, 2023):
        pre_processed_graphs[year] = {}
        output[year] = {}
        for month in range(1, 13):
            if year == 2022 and month > 10:
                continue
            G = station_networks[year][month]
            G = nx.DiGraph(G.subgraph(max(nx.weakly_connected_components(G), key = len)))
            
            sinks = []
            for node in G:
                if G.out_degree(node) == 0:
                    sinks.append(node)
                    in_neighbors = []
                    for node_j in G:
                        if node in G.neighbors(node_j):
                            in_neighbors.append(node_j)
                    for in_neighbor in in_neighbors:
                        G.add_edge(node, in_neighbor, weight = 1)
                                            
            top_race = {}
            num_na = 0
            for node_i in G:
                total_trips = np.sum([G[node_i][node_j]["weight"] for node_j in G.neighbors(node_i)])
                for node_j in G.neighbors(node_i):
                    G[node_i][node_j]["weight"] /= total_trips
                
                is_set = False
                if "station_id" in G.nodes[node_i]:
                    node_i_id = G.nodes[node_i]["station_id"]
                    if node_i_id in station_census_df.index:
                        top_race[node_i] = utils.largest_category(station_census_df, node_i_id, races, prefix = "Percentage ")
                        is_set = True
                        
                if not is_set:
                    top_race[node_i] = "NA"
                    num_na += 1
                
            nx.set_node_attributes(G, top_race, "Race")
            G_null = get_null_graph(G, "Race")
            pre_processed_graphs[year][month] = {"G": G, "G_null": G_null}
            output[year][month] = {"Sinks": sinks,
                                  "Number of Stations": len(G),
                                  "Number of Unlabled Stations": num_na}
    
#     fig, ax = plt.subplots()
#     counts = [output[year][month] for month in output[year] for year in output]
#     output_df = pd.DataFrame(counts).set_index("Date")
#     output_df["Number of Stations"].plot(ax=ax)
#     output_df["Number of Unlabled Stations"].plot(ax=ax)
#     plt.savefig("station_counts.pdf", bbox_inches="tight")
            
#     for year in [2019]:
#         output_year = metrics_per_year(pre_processed_graphs, year, "Race", races)
#         for month in output_year:
#             for key in output_year[month]:
#                 output[year][month][key] = output_year[month][key]
                
#     print(output)
    
    job_info = {}
    jobs = []

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(timeout_min=int(7.75 * 60),
                               cpus_per_task=24,
                               slurm_partition="short")

    for year in range(2015, 2023):
        job = executor.submit(metrics_per_year,
                             pre_processed_graphs,
                             year,
                             "Race",
                             races)
        job_info[job.job_id] = {"year": year}
        print("year: {} job id: {}".format(year, job.job_id))
        jobs.append(job)

    for job in jobs:
        output_year = job.result()
        year = job_info[job.job_id]["year"]
        for month in output_year:
            for key in output_year[month]:
                output[year][month][key] = output_year[month][key]
    
        with open("intermediates/segregation_by_month_parallel.pickle", "wb") as pickleFile:
            pickle.dump(output, pickleFile)