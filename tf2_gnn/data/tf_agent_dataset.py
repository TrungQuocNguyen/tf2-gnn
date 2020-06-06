from typing import Any, Dict, Iterable, List, Iterator, Tuple, Optional, Set

import numpy as np
import tensorflow as tf
from dpu_utils.utils import RichPath

from .graph_dataset import DataFold, GraphSample, GraphBatchTFDataDescription, GraphDataset
from .utils import compute_number_of_edge_types, get_tied_edge_types, process_adjacency_lists

class TFAgentsGraphSample(GraphSample): 
    """Data structure holding a single TF agents graph."""
    def __init__(
        self,
        adjacency_lists: List[np.ndarray],
        type_to_node_to_num_inedges: np.ndarray,
        node_features: np.ndarray,
        node_targets: np.ndarray,
    ):
        super().__init__(adjacency_lists, type_to_node_to_num_inedges, node_features)
        self._node_targets = node_targets
    @property
    def node_targets(self) -> np.ndarray:
        """Node targets to predict regression values for each node. 
        Size [V, C], V: number of node, C: number of regression values to predict e.g: steering, acceleration"""
        return self._node_targets

class TFAgentsDataset(GraphDataset[TFAgentsGraphSample]):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "max_nodes_per_batch": 240,
            "add_self_loop_edges": False,
            "tie_fwd_bkwd_edges": True,
        }
    @staticmethod
    def default_data_path() -> str:
        return "data"

    def __init__(self, params: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        super().__init__(params, metadata=metadata)

        self._tied_fwd_bkwd_edge_types = get_tied_edge_types(
            tie_fwd_bkwd_edges=params["tie_fwd_bkwd_edges"], num_fwd_edge_types=1,
        )

        self._num_edge_types = compute_number_of_edge_types(
            tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            num_fwd_edge_types=1,
            add_self_loop_edges=params["add_self_loop_edges"],
        )

        # Things that will be filled once we load data:
        self._loaded_data: Dict[DataFold, List[TFAgentsGraphSample]] = {}
    
    @property
    def num_edge_types(self) -> int:
        return self._num_edge_types

    @property
    def node_feature_shape(self) -> Tuple:
        some_data_fold = next(iter(self._loaded_data.values()))
        return (some_data_fold[0].node_features.shape[-1],)

    @property
    def num_node_regression_targets(self) -> int:
        return 2

    
    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath, folds_to_load: Optional[Set[DataFold]] = None) -> None:
        # Data in format as downloaded from https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip
        # If we haven't defined what folds to load, load all:
        if folds_to_load is None:
            folds_to_load = {DataFold.TRAIN, DataFold.VALIDATION, DataFold.TEST}

        if DataFold.TRAIN in folds_to_load:
            self._loaded_data[DataFold.TRAIN] = self.__load_data(path, DataFold.TRAIN)
        if DataFold.VALIDATION in folds_to_load:
            self._loaded_data[DataFold.VALIDATION] = self.__load_data(path, DataFold.VALIDATION)
        if DataFold.TEST in folds_to_load:
            self._loaded_data[DataFold.TEST] = self.__load_data(path, DataFold.TEST)

    def load_data_from_list(
        self, datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST
    ):
        if target_fold == DataFold.TRAIN: 
             self._loaded_data[DataFold.TRAIN] = self.__load_data_from_list(datapoints, DataFold.TRAIN)
        if target_fold == DataFold.VALIDATION: 
             self._loaded_data[DataFold.VALIDATION] = self.__load_data_from_list(datapoints, DataFold.VALIDATION)
        if target_fold == DataFold.TEST: 
             self._loaded_data[DataFold.TEST] = self.__load_data_from_list(datapoints, DataFold.TEST)

    def __load_data_from_list(
        self,  datapoints: List[Dict[str, Any]], target_fold: DataFold = DataFold.TEST): 

        graph_id_to_edges: Dict[int, List[Tuple[int, int]]] = {}
        graph_id_to_features: Dict[int, List[np.ndarray]] = {}
        graph_id_to_targets: Dict[int, List[np.ndarray]] = {}
        #graph_id_to_node_offset: Dict[int, int] = {}

        for graph_id, graph in enumerate(datapoints, 1):    #datapoints is a list of dictionary. graph_id start at 1
            graph_id_to_features[graph_id] = []
            graph_id_to_targets[graph_id] = []
            graph_id_to_edges[graph_id] = []
            for node_dict in graph['graph']['nodes']: 
                graph_id_to_features[graph_id].append(np.array(list(node_dict.values())))
            for target_dict in list(graph['actions'].values()): 
                graph_id_to_targets[graph_id].append(np.array(list(target_dict.values())))
            #convert source, target node from id to position of nodes in the list 
            for edge_dict in graph['graph']['links']: 
                src_node, tgt_node  = edge_dict['source'], edge_dict['target']
                graph_id_to_edges[graph_id].append((src_node, tgt_node))

        
        
        final_graphs = []
        for graph_id in graph_id_to_edges.keys():
            num_nodes = len(graph_id_to_features[graph_id])

            adjacency_lists, type_to_node_to_num_inedges = process_adjacency_lists(
                adjacency_lists=[graph_id_to_edges[graph_id]],
                num_nodes=num_nodes,
                add_self_loop_edges=self.params["add_self_loop_edges"],
                tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            )

            final_graphs.append(
                TFAgentsGraphSample(
                    adjacency_lists=adjacency_lists,
                    type_to_node_to_num_inedges=type_to_node_to_num_inedges,
                    node_features=np.array(graph_id_to_features[graph_id]),
                    node_targets=np.array(graph_id_to_targets[graph_id]),
                )
            )

        return final_graphs
        

    def __load_data(self, data_dir: RichPath, data_fold: DataFold):
        if data_fold == DataFold.TRAIN:
            data_name = "train"
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
        elif data_fold == DataFold.TEST:
            data_name = "test"
        else:
            raise ValueError("Unknown data fold '%s'" % str(data_fold))
        print(" Loading observer %s data from %s." % (data_name, data_dir))
        
        graph_pickle_data = data_dir.join("%s_data_collection.pickle" % data_name).read_by_file_suffix()

        graph_id_to_edges: Dict[int, List[Tuple[int, int]]] = {}
        graph_id_to_features: Dict[int, List[np.ndarray]] = {}
        graph_id_to_targets: Dict[int, List[np.ndarray]] = {}
        #graph_id_to_node_offset: Dict[int, int] = {}

        for graph_id, graph in enumerate(graph_pickle_data, 1):    #graph_pickle_data is a list of dictionary. graph_id start at 1
            graph_id_to_features[graph_id] = []
            graph_id_to_targets[graph_id] = []
            graph_id_to_edges[graph_id] = []
            for node_dict in graph['graph']['nodes']: 
                graph_id_to_features[graph_id].append(np.array(list(node_dict.values())))
            for target_dict in graph['actions'].values(): 
                graph_id_to_targets[graph_id].append(np.array(list(target_dict.values())))
            #convert source, target node from id to position of nodes in the list 
            for edge_dict in graph['graph']['links']: 
                src_node, tgt_node  = edge_dict['source'], edge_dict['target']
                graph_id_to_edges[graph_id].append((src_node, tgt_node))

        
        
        final_graphs = []
        for graph_id in graph_id_to_edges.keys():
            num_nodes = len(graph_id_to_features[graph_id])

            adjacency_lists, type_to_node_to_num_inedges = process_adjacency_lists(
                adjacency_lists=[graph_id_to_edges[graph_id]],
                num_nodes=num_nodes,
                add_self_loop_edges=self.params["add_self_loop_edges"],
                tied_fwd_bkwd_edge_types=self._tied_fwd_bkwd_edge_types,
            )

            final_graphs.append(
                TFAgentsGraphSample(
                    adjacency_lists=adjacency_lists,
                    type_to_node_to_num_inedges=type_to_node_to_num_inedges,
                    node_features=np.array(graph_id_to_features[graph_id]),
                    node_targets=np.array(graph_id_to_targets[graph_id]),
                )
            )

        return final_graphs

        
        #return (graph_id_to_features, graph_id_to_targets,graph_id_to_edges, graph_pickle_data)
    def get_batch_tf_data_description(self) -> GraphBatchTFDataDescription:
        data_description = super().get_batch_tf_data_description()
        return GraphBatchTFDataDescription(
            batch_features_types=data_description.batch_features_types,
            batch_features_shapes=data_description.batch_features_shapes,
            batch_labels_types={**data_description.batch_labels_types, "node_targets": tf.float32},   # the name "node_targets" must match with that name in _new_batch, add_graph_to_batch, finalise_batch
            batch_labels_shapes={**data_description.batch_labels_shapes, "node_targets": (None, None)},
        )

    def _graph_iterator(self, data_fold: DataFold) -> Iterator[TFAgentsGraphSample]:
        loaded_data = self._loaded_data[data_fold]#loaded_data is a list of TFAgentsGraphSample
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(loaded_data)   #shuffle the position of TFAgentsGraphSample in the list
        return iter(loaded_data) #return a iterator

    def _new_batch(self) -> Dict[str, Any]:
        new_batch = super()._new_batch()
        new_batch["node_targets"] = []

        return new_batch

    def _add_graph_to_batch(self, raw_batch, graph_sample: TFAgentsGraphSample) -> None:
        super()._add_graph_to_batch(raw_batch, graph_sample)
        raw_batch["node_targets"].append(graph_sample.node_targets)

    def _finalise_batch(self, raw_batch) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        batch_features, batch_targets = super()._finalise_batch(raw_batch)
        batch_targets["node_targets"] = np.concatenate(raw_batch["node_targets"], axis=0)
        return batch_features, batch_targets   

        
