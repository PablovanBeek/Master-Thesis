from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
import pandas as pd
import igraph as ig
from graphrole import RecursiveFeatureExtractor, RoleExtractor
import numpy as np
import time
from datetime import datetime, timedelta
import os
import json
import random

class Preprocessing():
    def __init__(self, dataset_name, training_days, test_days, amount_of_graphs) -> None:
        """
        Creates the instance of the class

        Parameters:
            dataset_name: Name of the TGBL dataset that needs to be analysed.
            training_days: The amount of data in days that one single training graph has.
            test_days: The amount of data in days that the test graph has.
            amount_of_graphs: The amount of training graphs that are used.

        Returns:
            Instance of the class.
        """
        self.dataset_name = dataset_name
        self.training_days = training_days
        self.test_days = test_days
        self.amount_of_graphs = amount_of_graphs
        self.graphs_dict = {}
        self.start_time = time.time()
        self.df = None
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        return

    def load_and_process_data(self) -> None:
        """
        Processes a dataset from the PyGLinkPropPredDataset class.
        It converts the timestamp column from unix to Pandas DataTime.

        Parameters:  
            Self: the instances of the class.

        Returns: 
            None: The DataFrame is assigned to self.df
        """

        dataset = PyGLinkPropPredDataset(name=self.dataset_name, root="datasets")
        data = dataset.get_TemporalData()
        
        data_dict = {
            "source": data.src.numpy(),
            "destination": data.dst.numpy(),
            "timestamp": data.t.numpy(),
            "y": data.y.numpy()
        }
        
        df = pd.DataFrame(data_dict)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        # Comment below is used for sampling
        # df = df.groupby(df['timestamp'].dt.date).sample(n=100)
        self.df = df
        return   

    def sample(self) -> None:    
        """
        Processes the DataFrame using sliding windows.
        The training and testing data are stored in a dict together with their masks.

        Parameters:  
            Self: the instances of the class.

        Returns: 
            None: The graphs_dict is assigned to to the class object.
                  The graph dict is populated with the training & test data.
                  Both the training and test data have masks.
        """
        # Create sliding window where each graph itteration shifts by one day
        for count in range(self.amount_of_graphs):
            # Start training at the beginning of the dataset
            start_train = self.df['timestamp'].min() + pd.Timedelta(days=count)
            end_train = start_train + pd.Timedelta(days=self.training_days)
            start_mask = end_train
            end_mask = start_mask + pd.Timedelta(days=(self.training_days + 3))
            
            # Do the acutal sampling of the training data
            train_data = self.df[(self.df['timestamp'] >= start_train) & (self.df['timestamp'] <= end_train)]
            
             # Do the acutal sampling of the mask data
            train_mask_data = self.df[(self.df['timestamp'] >= start_mask) & (self.df['timestamp'] <= end_mask)]

            # Assign to dict
            self.graphs_dict[f"train {count}"] = {'train': train_data, 'mask': train_mask_data}
       
        start_test = end_mask
        end_test = start_test + pd.Timedelta(days=self.test_days)
        
        # Test data sampling
        test_data = self.df[(self.df['timestamp'] >= start_test) & (self.df['timestamp'] <= end_test)]
    
        # Test mask sampling
        start_test_mask = end_test
        end_test_mask = start_test_mask + pd.Timedelta(days=self.training_days + 3)
        test_mask_data = self.df[(self.df['timestamp'] >= start_test_mask) & (self.df['timestamp'] <= end_test_mask)]

        self.graphs_dict['test'] = {'test': test_data, 'mask': test_mask_data}
        return


    def create_graphs(self) -> None:
        """
        Create N  graphs based on N datasets.
        Parameters:  
            Self: the instances of the class.
        Returns: 
            None: The graphs are assigned to self.graphs_dict.    
        """
        for key, graph_data in self.graphs_dict.items():
            # Get training edges
            try :
                train_edges = list(zip(graph_data['train']['source'], graph_data['train']['destination'], graph_data['train']['timestamp']))

                # Create graph from training edges
                graph = ig.Graph.TupleList(train_edges, directed=True, edge_attrs=['timestamp'])
            except:
                test_edges = list(zip(graph_data['test']['source'], graph_data['test']['destination'], graph_data['test']['timestamp']))
            
                # Create graph from testing edges
                graph = ig.Graph.TupleList(test_edges, directed=True, edge_attrs=['timestamp'])

            # Storing the graph in a dictionary
            self.graphs_dict[key]['graph'] = graph
        return

    def role_mining(self) -> None:
        """
        Calculates the roles for all of the nodes.
        Parameters:  
            Self: the instances of the class.
        Returns: 
            None: A pandas DataFrame is assigned to the self.graphs_dict for each training and test networks
        """
        for key, graph_data in self.graphs_dict.items():
            # Network from dictonary
            g = graph_data['graph']
            
            feature_extractor = RecursiveFeatureExtractor(g)
            features = feature_extractor.extract_features()

            role_extractor = RoleExtractor(n_roles=None)
            role_extractor.extract_role_factors(features)
            
            # Create DataFrame from roles and set node IDs as index
            node_id = g.vs["name"]
            roles_df = role_extractor.role_percentage
            roles_df['node'] = node_id

            graph_data["roles_df"] = roles_df
        return
    
    def node_features(self) -> None:
        """
        Calculates features for all of the nodes.
        Parameters:  
            Self: the instances of the class.
        Returns: 
            None: A pandas DataFrame is assigned to the self.graphs_dict for each training and test networks
        """
        for key, graph_data in self.graphs_dict.items():
            # Network from dictonary
            g = graph_data['graph']
            
            pagerank = g.pagerank()
            indegree = g.indegree()
            outdegree = g.outdegree()
            eigenvector = g.eigenvector_centrality(directed=True)
            avg_path_length = g.average_path_length(directed=True)
            
            features_df = pd.DataFrame({
                'node': g.vs['name'],
                'pagerank': pagerank,
                'indegree': indegree,
                'outdegree': outdegree,
                'eigenvector': eigenvector,
                'avg_path_length': avg_path_length
            })
            
            graph_data['features_df'] = features_df
        return
    
    def mask_to_edges(self) -> None:
        """
        It filters out all of the nodes that are not in the graph but are present in the sampled mask
            Translates the original mask to the following DataFrame:
            Source: The source node
            destination: the destination node
            timestamp: The timestamp
            link: boolean value if the link valid or invalid
        Parameters:  
            Self: the instances of the class.
        Returns: 
            None: A pandas DataFrame is assigned to the self.graphs_dict for each training and test networks
        """
        for key, graph_data in self.graphs_dict.items():          
            g = graph_data['graph']
            
            nodes = list(g.vs['name'])

            df_mask = pd.DataFrame(graph_data["mask"])
            df_mask.drop("y", inplace=True, axis=1)
            df_mask["link"] = 1

            # Remove the nodes from the mask if the node are not in the graph
            df_mask = df_mask[df_mask['source'].isin(nodes) & df_mask['destination'].isin(nodes)]
            graph_data['mask'] = df_mask
        return

    def populate_dataframe(self):
        """
        Combines the different dataframes for each graph.
        Adds the graph_key column
        The Roles and Features get added to the DataFrame that is generated in mask_to_edges()
        Parameters:  
            Self: the instances of the class.
        Returns: 
            None: A pandas DataFrame for training is assigned as self.train
                  A pandas Dataframe for testing is assigned as self.test
        """
        for key, graph_data in self.graphs_dict.items():
            data = graph_data["mask"]
    
            # Prefixes for the column names after merging
            source_features = (graph_data['features_df']).add_prefix("source_")
            source_roles = (graph_data['roles_df']).add_prefix("source_")
            destination_features = (graph_data['features_df']).add_prefix("destination_")
            destination_roles = (graph_data['roles_df']).add_prefix("destination_")
            
            # Merging the role and feature dataframes
            self.source_roles_features = pd.merge(source_features, source_roles, on="source_node", how='left', validate='1:1')
            self.destination_roles_features = pd.merge(destination_features, destination_roles, on="destination_node", how='left', validate='1:1')
            
            # Merging the mask with the roles and features
            data = pd.merge(data, self.source_roles_features, left_on='source', right_on='source_node', how='left', validate='m:1')
            data = pd.merge(data, self.destination_roles_features, left_on='destination', right_on='destination_node', how='left', validate='m:1')
            
            # Drop the duplicated columns
            data.drop(["source_node", "destination_node"], axis=1, inplace=True)

            # Add the graph key
            data["graph_key"] = key

            # Merge data with the datasets that have already been merged
            if 'train' in graph_data:
                self.train = pd.concat([self.train, data], ignore_index=True)
            elif "test" in graph_data:  
                self.test = pd.concat([self.test, data], ignore_index=True)
        return

    def add_invalid_edges(self) -> None:
        """
        Adds edges that do not exist to the DataFrame.
        Both edges need to be in the same graph.
        It creates as many invalid transactions as there are valid transactions.
        The invalid edges are not used for the test graphs.

        Parameters:  
            Self: the instances of the class.

        Returns: 
            None: A pandas DataFrame for training is assigned as self.train
                  A pandas Dataframe for testing is assigned as self.test
        """
        # Dict to be able to store the datasets within the loop
        datasets = {'train': self.train, 'test': self.test}

        for key, dataset in datasets.items():
            source_columns = []
            destination_columns = []

            # Loop over all of the columns in the dataset
            # This makes it possible to subset  the dataset in source and destination
            for column in dataset.columns:
                if column.startswith("source"):
                    source_columns.append(column)
                if column.startswith("destination"):
                    destination_columns.append(column)

            source_df = dataset[source_columns]
            destination_df = dataset[destination_columns]

            # Drop duplicate values in the source and destination DataFrame
            source_df = source_df.drop_duplicates(subset=['source'])
            destination_df = destination_df.drop_duplicates(subset=['destination'])

            sources = dataset["source"].to_list()
            destinations = dataset["destination"].to_list()
            timestamps = dataset["timestamp"].to_list()

            pairs = set()
            random_pairs = set()
            
            # Add all of the existing edges to a set
            for index, row in dataset.iterrows():
                pairs.add((row['source'], row['destination']))

            while len(random_pairs) < len(dataset):
                random_source = random.choice(sources)
                random_destination = random.choice(destinations)
                # Check if link does not exist already
                if (random_source, random_destination) not in pairs and (random_source, random_destination) not in random_pairs:
                     for graph_key, graph_data in self.graphs_dict.items():
                        g = graph_data["graph"]
                        # Check if both nodes are available in the correct graph type
                        if (random_source in g.vs['name']) and (random_destination in g.vs['name']) and (key in graph_key):
                            random_pairs.add((random_source, random_destination, graph_key))
                            break
                
            
            random_pairs_df = pd.DataFrame(list(random_pairs), columns=["source", "destination", "graph_key"])
            
            # Merge the features with the created invalid edges
            random_pairs_df = random_pairs_df.merge(source_df, how='left', on='source')
            random_pairs_df = random_pairs_df.merge(destination_df, how='left', on='destination')

            # Add the other columns 
            random_pairs_df["link"] = 0
            random_pairs_df["timestamp"] = np.random.choice(timestamps, size=len(random_pairs_df), replace=True)

            # Concat the new data to the original data
            if key == "train":
                self.train = pd.concat([self.train, random_pairs_df], ignore_index=True)
            elif key == "test":
                # Do not assign the false edges to the dataframe
                self.test = self.test

    def node_pair_features(self):
        # Initialize lists with None values for all rows
        jaccard_train = [None] * len(self.train)
        sorensen_train = [None] * len(self.train)

        jaccard_test = [None] * len(self.test)
        sorensen_test = [None] * len(self.test)

        datasets = {'train': self.train, 'test': self.test}
        for key, dataset in datasets.items():
            for id, row in dataset.iterrows():
                source = int(row["source"])
                destination = int(row['destination'])

                jaccard = None
                sorensen = None

                for graph_key, graph_data in self.graphs_dict.items():
                    g = graph_data['graph']
                    if source in g.vs['name'] and destination in g.vs['name']:
                        try:
                            # Convert node names to internal vertex IDs
                            source_id = g.vs.find(name=source).index
                            destination_id = g.vs.find(name=destination).index
                            
                            # Calculate Jaccard coefficient
                            jaccard = g.similarity_jaccard(pairs=[(source_id, destination_id)])[0]

                            # Calculate Sorensen similarity
                            sorensen = g.similarity_dice(pairs=[(source_id, destination_id)])[0]
                            break
                        except ig._igraph.InternalError:
                            continue

                if jaccard is not None and sorensen is not None:
                    if key == "train":
                        jaccard_train[id] = jaccard
                        sorensen_train[id] = sorensen
                    else:
                        jaccard_test[id] = jaccard
                        sorensen_test[id] = sorensen

        self.train["jaccard"] = jaccard_train
        self.train["sorensen"] = sorensen_train

        self.test["jaccard"] = jaccard_test
        self.test["sorensen"] = sorensen_test
        return

    def save_config_data(self):
        """  
        Saves the training, test and config data to different files.

        Parameters:  
            Self: the instances of the class.

        Returns: 
            None
        """             
        run_time = time.time() - self.start_time
        hms = str(timedelta(seconds=run_time))

        current_datetime = datetime.now().strftime("%m_%d_%Y, %H_%M_%S")
        folder_path = os.path.join("saved_data", current_datetime)
        os.makedirs(folder_path, exist_ok=True)

        self.train.to_csv(os.path.join(folder_path, "train.csv"), index=False)
        self.test.to_csv(os.path.join(folder_path, "test.csv"), index=False)

        info = {
            "datetime": current_datetime,
            "training_days": self.training_days,
            "test_days": self.test_days,
            "amount_of_graphs": self.amount_of_graphs,
            "columns": self.train.columns.tolist(),
            "Hours:minutes:seconds": hms,
        }

        # Save the config
        with open(os.path.join(folder_path, "config.json"), "w") as json_file:
            json.dump(info, json_file)

        print(f"Data and config saved in folder: {folder_path}")
        return

    def analyse(self):
        self.load_and_process_data()
        self.sample()
        self.create_graphs()
        self.mask_to_edges()
        self.role_mining()
        self.node_features()
        self.populate_dataframe()
        self.add_invalid_edges()
        self.node_pair_features()
        self.save_config_data()
        print("The process has finished")
        return 

preprocessor = Preprocessing("tgbl-coin", training_days=3, test_days=3, amount_of_graphs=3)
preprocessor.analyse()

