"""Causal models."""

# Copyright 2024 Anil Rao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import numpy as np
from scipy.stats import norm, bernoulli
import pandas as pd

from .exceptions import MissingNodeError
from .node_models import NodeAdditiveNoiseModel


def samples_df_to_dict(samples_df):
    """Convert samples pandas dataframe to samples dictionary."""
    samples_data_as_lists = samples_df.to_dict('list')
    samples_dict = {
        variable_name: np.array(variable_data) for variable_name, variable_data in samples_data_as_lists.items()
    }
    return samples_dict


class StructuralCausalModel:
    """Structural Causal Model in which every model of the causal mechanism of a variable/node in corresponding DAG,
    is of a specific form.

    Parameters
    ----------
    node_models: list
        Each entry in the list is a member of a subclass of NodePostTransformedPolynomialModel.
    """
    def __init__(self, node_models):
        self.node_models = node_models
        self._sort_node_models()

    def _sort_node_models(self):
        """Topologically sort nodes in the DAG."""
        # firstly sort the nodes alphabetically
        node_models = copy.deepcopy(self.node_models)
        node_names = [node_model.name for node_model in node_models]
        node_names_arg_sort = np.argsort(node_names)
        alphabetically_sorted_node_models = [node_models[idx] for idx in node_names_arg_sort]

        # now need to work out the topological order
        # we do this by iteratively testing if a node's parents are in current sorted list
        # if they are, we add to the sorted list

        # initialize list of topologically sorted nodes
        topologically_sorted_node_models = []
        # keep track of their names
        topologically_sorted_node_names = []

        number_of_nodes = len(node_models)

        while len(topologically_sorted_node_models) < number_of_nodes:
            # if all nodes haven't been sorted, test the unsorted ones
            # record number of nodes sorted this round of tests
            number_nodes_sorted_this_round = 0

            # loop through nodes for this round
            for node_model in alphabetically_sorted_node_models:
                # test unsorted nodes
                if node_model.name not in topologically_sorted_node_names:
                    # test if node's parents have already been sorted
                    if set(node_model.parent_variables).issubset(set(topologically_sorted_node_names)):
                        # update sorted lists and counter
                        topologically_sorted_node_models.append(node_model)
                        topologically_sorted_node_names.append(node_model.name)
                        number_nodes_sorted_this_round += 1

            # raise error if we didn't sort any nodes this round
            if number_nodes_sorted_this_round == 0:
                # print names of nodes that cannot be sorted
                for node_name in node_names:
                    if node_name not in topologically_sorted_node_names:
                        print(f"variable {node_name} cannot be sampled from this SCM")
                raise RuntimeError('SCM does not have valid DAG')

        self.node_models = topologically_sorted_node_models

    def draw_sample(self, size=1000, initial_random_state=0, return_dataframe=True):
        """Draw sample from the scm.

        Parameters
        ----------
        size: int, default=1000
            Size of sample.
        initial_random_state: int, default=0
            Random state used by u_draw_random_variates of the first node.
            The random state increments by one before each draw from
            subsequent nodes.
        return_dataframe: boolean, default=True
            Flag which when true will return a pandas dataframe otherwise a dictionary.

        Returns
        -------
        scm_samples: pandas dataframe or dict of str: numpy array
            Contains drawn samples for each variable of the scm.
            If `return_dataframe` is True, the dataframe has indexes 0 to `size`-1
            and each column contains the sample for the variable with name column name.
            If `return_dataframe` is False, dictionary contains
            variable name mapped to sample of size (`size`,).
        """

        scm_samples = {}
        # draw sample for each variable in order
        print("Drawing from...")
        for idx, node_model in enumerate(self.node_models):
            print(f"{node_model.name}")
            scm_samples[node_model.name] = node_model.draw_sample(
                size=size,
                random_state=idx + initial_random_state,
                parent_data=scm_samples
            )
        print("done")
        # return dataframe or dictionary of samples as required
        if return_dataframe:
            scm_samples = pd.DataFrame.from_dict(scm_samples)

        return scm_samples

    def _verify_node_in_scm(self, node_name):
        """Verify node is part of scm.

        Parameters
        ----------
        node_name: str
            Name of variable/node which we are checking in scm.
        """
        scm_node_names = [node_model.name for node_model in self.node_models]
        if node_name not in scm_node_names:
            raise MissingNodeError(f"variable {node_name} not part of SCM. SCM has variables {', '.join(scm_node_names)}")

    def draw_rct_sample(self, size=1000, initial_random_state=0, return_dataframe=True, treatment_variable='X',
                        intervention_draw_random_variates=bernoulli.rvs, **i_kwargs):
        """Draw rct sample from the scm.

        Parameters
        ----------
        size: int
            Size of sample.
        initial_random_state: int, default=0
            Random state used by u_draw_random_variates of the first variable in the StructuralCausalModel object.
            The random state increments by one before each draw from subsequent variables.
        return_dataframe: boolean, default=True
            Flag which when true will return a pandas dataframe otherwise a dictionary.
        treatment_variable: str, default='X'
            Name of variable which we are intervening on.
        intervention_draw_random_variates: method, default=scipy.stats.bernoulli.rvs
            Method that draws random variates from the intervention distribution of treatment variable.
            Expected to be an rvs method of an implemented or custom distribution from scipy.stats.
        **i_kwargs: optional
            Distribution parameters of `intervention_draw_random_variates`.

        Returns
        -------
        pandas dataframe or dict of str: numpy array
            Contains drawn samples for each variable in the interventional scm.
            If `return_dataframe` is True, the dataframe has indexes 0 to `size`-1
            and each column contains the sample for the variable with name column name.
            If `return_dataframe` is False, dictionary contains
            variable name mapped to sample of size (`size`,).
        """
        # check that treatment_variable exists in the scm
        self._verify_node_in_scm(treatment_variable)
        # create intervention node model
        intervention_node_model = NodeAdditiveNoiseModel(treatment_variable, u_draw_random_variates=intervention_draw_random_variates, **i_kwargs)
        # deep copy the original scm so that original scm is untouched
        intervention_scm = copy.deepcopy(self)
        # replace the treatment variable node model with the intervention node model
        for idx, node_model in enumerate(intervention_scm.node_models):
            if node_model.name == treatment_variable:
                intervention_scm.node_models[idx] = intervention_node_model
        # draw the sample from the intervention scm
        # note that all nodes from the intervention scm, even disconnected ones, will be sampled
        return intervention_scm.draw_sample(
            size=size, initial_random_state=initial_random_state, return_dataframe=return_dataframe
        )

    def draw_do_operator_sample(self, size=1000, initial_random_state=0, return_dataframe=True,
                                intervention_variable='X', intervention_value=1.):
        """Draw do-operator sample from the scm.

        Parameters
        ----------
        size: int
            Size of sample
        initial_random_state: int, default=0
            Random state used by u_draw_random_variates of the first variable in the StructuralCausalModel object.
            The random state increments by one before each draw from subsequent variables.
        return_dataframe: boolean, default=True
            Flag which when true will return a pandas dataframe otherwise a dictionary.
        intervention_variable: str, default='X'
            Name of variable we are do-intervening on.
        intervention_value: float, default=1.
            Do-intervention value.

        Returns
        -------
        pandas dataframe or dict of str: numpy array
            Contains drawn samples for each variable in the interventional scm.
            If `return_dataframe` is True, the dataframe has indexes 0 to `size`-1
            and each column contains the sample for the variable with name column name.
            If `return_dataframe` is False, dictionary contains
            variable name mapped to sample of size (`size`,).
        """
        # we operationalise do-operator sampling as a trial in which every subject receives the same level of treatment
        # we therefore call the draw_rct_sample method using an intervention distribution with corresponding mean
        # and zero variance
        return self.draw_rct_sample(
            size=size, initial_random_state=initial_random_state, return_dataframe=return_dataframe,
            treatment_variable=intervention_variable,
            intervention_draw_random_variates=norm.rvs, loc=intervention_value, scale=0.
        )

    def compute_counterfactuals(self, observed_data, intervention_variable='X', intervention_values=None,
                                return_dataframe=True):
        """Compute counterfactuals for variables in the scm.

        Parameters
        ----------
        observed_data: dataframe
            Observed data samples for each node in scm.
        intervention_variable: str, default='X'
            Name of variable which is being intervened on under the counterfactual.
        intervention_values: 1-d numpy array, optional
            Values of `intervention_variable` under the counterfactual. Default is a numpy array of ones.
        return_dataframe: boolean, default=True
            Flag which when true will return a pandas dataframe otherwise a dictionary.

        Returns
        -------
        computed_counterfactuals: pandas dataframe or dict of str: numpy array
            Contains computed counterfactuals for each variable of scm.
            If `return_dataframe` is True, each dataframe column contains the computed counterfactuals
            for the variable with name column name.
            If `return_dataframe` is False, dictionary contains
            variable name mapped to computed counterfactuals of that variable.
        """
        # parse intervention_values
        if intervention_values is None:
            intervention_values = np.ones(shape=observed_data.shape[0])

        # check that intervention_variable exists in the scm
        self._verify_node_in_scm(intervention_variable)

        # convert dataframe into dictionary ignoring indexes
        observed_data_as_dict = samples_df_to_dict(observed_data)

        # create dictionary for storing all abducted exogenous data
        abducted_exogenous = {}
        # abduct exogenous data for each variable in order
        print("Abducting exogenous data for...")
        for node_model in self.node_models:
            # no need to abduct exogenous data for intervention variable
            if node_model.name != intervention_variable:
                # check whether the exogenous data for this variable can be abducted by inspecting the model for its causal mechanism

                if hasattr(node_model, 'abduct_exogenous') and callable(node_model.abduct_exogenous):
                    print(f"{node_model.name}")
                    abducted_exogenous[node_model.name] = node_model.abduct_exogenous(observed_data_as_dict)
                else:
                    # note that following is raised when abduction method is not implemented for this node model class
                    # it does not differentiate between a missing abduction method (for invertible post-transformation
                    # functions) and non-invertible post-transformation functions which require probabilistic abduction
                    raise RuntimeError(f"no implementation for abducting exogenous data of variable {node_model.name} which has a causal mechanism of model type {type(node_model).__name__}")

        # create dictionary for storing computed counterfactuals
        computed_counterfactuals = {intervention_variable: intervention_values}
        # predict each variable in order
        print(f"Predicting...")
        for node_model in self.node_models:
            if node_model.name != intervention_variable:
                print(f"{node_model.name}")
                computed_counterfactuals[node_model.name] = node_model.predict(computed_counterfactuals, abducted_exogenous)
        print("done")

        # return dataframe or dictionary of computed counterfactuals as required
        if return_dataframe:
            # create computed counterfactuals dataframe with same index as observed data dataframe
            computed_counterfactuals = pd.DataFrame(data=computed_counterfactuals, index=observed_data.index)

        return computed_counterfactuals

    def __str__(self):
        """Create string representation of the structural causal model."""
        scm_as_string = '\nStructural Causal Model\n=======================\n\n'
        for idx, node_model in enumerate(self.node_models, start=1):
            scm_as_string += str(node_model)
            if idx < len(self.node_models):
                scm_as_string += '\n\n'
        return scm_as_string
