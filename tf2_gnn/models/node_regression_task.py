from typing import Any, Dict, Iterable, List, NamedTuple, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf

from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphTaskModel


class NodeRegressionTask(GraphTaskModel):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {}
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
        if not hasattr(dataset, "num_node_regression_targets"):
            raise ValueError(f"Provided dataset of type {type(dataset)} does not provide num_node_regression_targets information.")
        self._num_regression_targets = dataset.num_node_regression_targets

    def build(self, input_shapes):
        with tf.name_scope(self.__class__.__name__):
            self.node_to_targets_layer = tf.keras.layers.Dense(units=self._num_regression_targets, use_bias=True)
            self.node_to_targets_layer.build((None, self._params["gnn_hidden_dim"]))
        super().build(input_shapes)

    def compute_task_output(
        self, batch_features, final_node_representations: tf.Tensor, training: bool
    ):
        per_node_logits = self.node_to_targets_layer(final_node_representations) #per_node_logits has size VxT: V is number of node, T is number of regression targets of each node. 
        return (per_node_logits,)

    def compute_task_metrics(
        self, batch_features, task_output, batch_labels
    ) -> Dict[str, tf.Tensor]:
        (per_node_logits,) = task_output
        ego_id = tf.cumsum(tf.pad(tf.unique_with_counts(batch_features["node_to_graph_map"])[2],[[1,0]]))[:-1]
        ego_logits = tf.gather(per_node_logits, ego_id)
        ego_labels = tf.gather(batch_labels["node_targets"], ego_id)
        #2 ways to compute the loss
        #   1.compute loss with respect to all nodes
        #loss = self._fast_task_metrics(per_node_logits, batch_labels["node_targets"])
        #   2.compute loss with respect to only ego nodes
        
        loss = self._fast_task_metrics(ego_logits,ego_labels)
        
        return {"loss": loss}

    @tf.function(input_signature=(tf.TensorSpec((None, None)), tf.TensorSpec((None, None))))
    def _fast_task_metrics(self, per_node_logits, node_targets):
        loss = tf.nn.l2_loss(per_node_logits-node_targets)
        return loss

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        avg_loss = np.average([r["loss"] for r in task_results])
        return avg_loss, f"Avg Loss per epoch: {avg_loss:.3f}"