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

import unittest

import numpy as np
from scipy import stats
import pandas as pd

from causalite import node_models as nm
from causalite import causal_models as cm


class TestCausalGraphicalModels(unittest.TestCase):
    def test_sort_nodes(self):
        """Test topological sorting of nodes in StructuralCausalModel object."""
        # create list and draw samples
        f_x = nm.NodeAdditiveNoiseModel('x', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_z = nm.NodeAdditiveNoiseModel('z')
        f_t = nm.NodeAdditiveNoiseModel('t', {'x': [1.], 'z': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'t': [0.2], 'x': [-1.], 'tx': [0.1], 'z': [-0.2]})
        f_b = nm.NodeBinaryLogisticModel('b', parent_polys={'z': [-1.2], 'x': [-5.6]})

        node_models = [f_b, f_t, f_y, f_x, f_z]
        scm = cm.StructuralCausalModel(node_models=node_models)
        scm_node_names = [node_model.name for node_model in scm.node_models]
        correct_ordering = ['x', 'z', 'b', 't', 'y']

        with self.subTest("Test if sorted ordering correct"):
            self.assertTrue(scm_node_names == correct_ordering)
        with self.subTest("Test if sorted node attributes have different id to original node attributes"):
            self.assertTrue(scm.node_models[0].parent_polys is not f_x.parent_polys)

    def test_draw_samples_scm(self):
        """Test drawing sample from StructuralCausalModel object."""
        f_x = nm.NodeAdditiveNoiseModel('x', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_t = nm.NodeBinaryLogisticModel('t', {'x': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'t': [0.2], 'x': [-1.], 'tx': [0.1], 'a': [-0.2]})

        scm = cm.StructuralCausalModel(node_models=[f_a, f_t, f_y, f_x])
        samples = scm.draw_sample(size=1000, initial_random_state=0)

        # draw samples manually and compare
        x = stats.randint.rvs(low=-1, high=2, size=1000, random_state=1).reshape(-1, 1)
        a = stats.norm.rvs(size=1000, random_state=0).reshape(-1, 1)
        t_latent_variable = x + 0.3 * a - 0.08 * pow(a, 2) + stats.logistic.rvs(size=1000, random_state=2).reshape(-1, 1)
        t = (t_latent_variable > 0.0) * 1.
        y = 0.2 * t - x + 0.1 * x * t - 0.2 * a + stats.norm.rvs(size=1000, random_state=3).reshape(-1, 1)
        samples_2 = pd.DataFrame(data=np.hstack((a, x, t, y)), columns=['a', 'x', 't', 'y'])

        self.assertTrue((samples - samples_2).abs().max().max() < 1e-10)

    def test_draw_samples_pure_interactions_scm(self):
        """Test drawing sample from StructuralCausalModel object where some parent polynomials are pure interactions."""
        f_x = nm.NodeAdditiveNoiseModel('x', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_t = nm.NodeAdditiveNoiseModel('t', {'ax': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'t': [0.2], 'x': [-1.], 'tx': [0.1], 'ax': [-0.2, 0.0, 4.1]})

        scm = cm.StructuralCausalModel(node_models=[f_a, f_t, f_y, f_x])
        samples = scm.draw_sample(size=1000, initial_random_state=0)

        # draw samples manually and compare
        x = stats.randint.rvs(low=-1, high=2, size=1000, random_state=1).reshape(-1, 1)
        a = stats.norm.rvs(size=1000, random_state=0).reshape(-1, 1)
        t = 0.3 * a * x - 0.08 * pow(a * x, 2) + stats.norm.rvs(size=1000, random_state=2).reshape(-1, 1)
        y = 0.2 * t - x + 0.1 * x * t - 0.2 * a * x + 4.1 * pow(a * x, 3) + stats.norm.rvs(size=1000, random_state=3).reshape(-1, 1)
        samples_2 = pd.DataFrame(data=np.hstack((a, x, t, y)), columns=['a', 'x', 't', 'y'])

        self.assertTrue((samples - samples_2).abs().max().max() < 1e-10)

    def test_simulate_rct_scm(self):
        """Test drawing sample from a simulated rct using StructuralCausalModel object."""
        f_c = nm.NodeAdditiveNoiseModel('c', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]})
        scm = cm.StructuralCausalModel(node_models=[f_a, f_c, f_x, f_y])
        samples = scm.draw_rct_sample(
            size=1000, initial_random_state=0,
            treatment_variable='x',
            p=0.5
        )
        # draw samples manually and compare
        c = stats.randint.rvs(low=-1, high=2, size=1000, random_state=1).reshape(-1, 1)
        a = stats.norm.rvs(size=1000, random_state=0).reshape(-1, 1)
        x = stats.bernoulli.rvs(size=1000, p=0.5, random_state=2).reshape(-1, 1)
        y = 0.2 * x - c + 0.1 * x * c - 0.2 * a + stats.norm.rvs(size=1000, random_state=3).reshape(-1, 1)
        samples_2 = pd.DataFrame(data=np.hstack((a, c, x, y)), columns=['a', 'c', 'x', 'y'])

        self.assertTrue((samples - samples_2).abs().max().max() < 1e-10)

    def test_simulate_rct_pure_interactions_scm(self):
        """Test drawing sample from StructuralCausalModel object where some parent polynomials are pure interactions."""
        f_x = nm.NodeAdditiveNoiseModel('x', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_t = nm.NodeAdditiveNoiseModel('t', {'ax': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'t': [0.2], 'x': [-1.], 'tx': [0.1], 'ax': [-0.2, 0.0, 4.1]})

        scm = cm.StructuralCausalModel(node_models=[f_a, f_t, f_y, f_x])
        samples = scm.draw_rct_sample(
            size=1000, initial_random_state=0,
            treatment_variable='t',
            p=0.5
        )
        # draw samples manually and compare
        x = stats.randint.rvs(low=-1, high=2, size=1000, random_state=1).reshape(-1, 1)
        a = stats.norm.rvs(size=1000, random_state=0).reshape(-1, 1)
        t = stats.bernoulli.rvs(size=1000, p=0.5, random_state=2).reshape(-1, 1)
        y = 0.2 * t - x + 0.1 * x * t - 0.2 * a * x + 4.1 * pow(a * x, 3) + stats.norm.rvs(size=1000, random_state=3).reshape(-1, 1)
        samples_2 = pd.DataFrame(data=np.hstack((a, x, t, y)), columns=['a', 'x', 't', 'y'])

        self.assertTrue((samples - samples_2).abs().max().max() < 1e-10)

    def test_do_operator_pure_interactions_scm(self):
        """Test drawing do-operator sample from StructuralCausalModel object where some parent polynomials are pure
        interactions.
        """
        f_x = nm.NodeAdditiveNoiseModel('x', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_t = nm.NodeAdditiveNoiseModel('t', {'ax': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'t': [0.2], 'x': [-1.], 'tx': [0.1], 'ax': [-0.2, 0.0, 4.1]})

        scm = cm.StructuralCausalModel(node_models=[f_a, f_t, f_y, f_x])
        samples = scm.draw_do_operator_sample(
            size=1000, initial_random_state=0,
            intervention_variable='t',
            intervention_value=3.5
        )
        # draw samples manually and compare
        x = stats.randint.rvs(low=-1, high=2, size=1000, random_state=1).reshape(-1, 1)
        a = stats.norm.rvs(size=1000, random_state=0).reshape(-1, 1)
        t = np.ones((1000, 1)) * 3.5
        y = 0.2 * t - x + 0.1 * x * t - 0.2 * a * x + 4.1 * pow(a * x, 3) + stats.norm.rvs(size=1000, random_state=3).reshape(-1, 1)
        samples_2 = pd.DataFrame(data=np.hstack((a, x, t, y)), columns=['a', 'x', 't', 'y'])

        self.assertTrue((samples - samples_2).abs().max().max() < 1e-10)

    def test_compute_counterfactuals_incorrect_model(self):
        """Test computation of counterfactuals for a StructuralCausalModel object.

        Here the model for the scm does not match that used to generate the observed data, as in the real world.
        """
        # generate observed data by defining an scm and drawing sample
        f_c_observed = nm.NodeAdditiveNoiseModel('c', {'a': [1.2]})
        f_a_observed = nm.NodeAdditiveNoiseModel('a')
        f_x_observed = nm.NodeAdditiveNoiseModel('x', {'c': [1.2], 'a': [0.35, -0.09]})
        f_y_observed = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'a': [-0.2]}, scale=5.)
        scm_observed = cm.StructuralCausalModel(node_models=[f_c_observed, f_a_observed, f_x_observed, f_y_observed])
        observed_samples = scm_observed.draw_sample(size=1000, initial_random_state=0)

        # define the scm model under which counterfactuals will be calculated
        # This scm model differs from the one used to generated observed data
        f_c = nm.NodeAdditiveNoiseModel('c', {'a': [1.5]})
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]}, scale=5.)
        scm = cm.StructuralCausalModel(node_models=[f_c, f_a, f_x, f_y])

        # compute counterfactuals for random values of c given the observed data under this scm
        computed_counterfactuals = scm.compute_counterfactuals(
            observed_samples,
            intervention_variable='c',
            intervention_values=stats.norm.rvs(size=1000, random_state=50)
        )

        # manually create counterfactuals given the observed data under this scm
        # we do this by abducting and predicting each node in turn apart from treatment node
        observed_samples_dict = cm.samples_df_to_dict(observed_samples)
        a_m = f_a.abduct_exogenous(observed_samples_dict)
        c_m = stats.norm.rvs(size=1000, random_state=50)
        x_m = c_m + 0.3 * a_m - 0.08 * a_m ** 2 + f_x.abduct_exogenous(observed_samples_dict)
        y_m = 0.2 * x_m - 1.0 * c_m + 0.1 * x_m * c_m - 0.2 * a_m + f_y.abduct_exogenous(observed_samples_dict)
        manual_counterfactuals = pd.DataFrame(
            data=np.column_stack((c_m, a_m, x_m, y_m)),
            columns=['c', 'a', 'x', 'y']
        )
        # compare manually created counterfactuals with computed counterfactuals
        self.assertTrue(np.abs(manual_counterfactuals - computed_counterfactuals).max().max() < 1e-10)

    def test_compute_counterfactuals_default_intervention_values_incorrect_model(self):
        """Test computation of counterfactuals for a StructuralCausalModel object.

        Here the model for the scm does not match that used to generate the observed data, as in the real world, and
        no intervention values are provided by user.
        """
        # generate observed data by defining an scm and drawing sample
        f_c_observed = nm.NodeAdditiveNoiseModel('c', {'a': [1.2]})
        f_a_observed = nm.NodeAdditiveNoiseModel('a')
        f_x_observed = nm.NodeAdditiveNoiseModel('x', {'c': [1.2], 'a': [0.35, -0.09]})
        f_y_observed = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'a': [-0.2]}, scale=5.)
        scm_observed = cm.StructuralCausalModel(node_models=[f_c_observed, f_a_observed, f_x_observed, f_y_observed])
        observed_samples = scm_observed.draw_sample(size=1000, initial_random_state=0)

        # define the scm model under which counterfactuals will be calculated
        # This scm model differs from the one used to generated observed data
        f_c = nm.NodeAdditiveNoiseModel('c', {'a': [1.5]})
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]}, scale=5.)
        scm = cm.StructuralCausalModel(node_models=[f_c, f_a, f_x, f_y])

        # compute counterfactuals for values of c given the observed data under this scm
        # do not provide intervention values to test if defaults are used correctly
        computed_counterfactuals = scm.compute_counterfactuals(
            observed_samples,
            intervention_variable='c'
        )

        # manually create counterfactuals given the observed data under this scm
        # we do this by abducting and predicting each node in turn apart from intervention node
        observed_samples_dict = cm.samples_df_to_dict(observed_samples)
        a_m = f_a.abduct_exogenous(observed_samples_dict)
        # create default intervention values
        c_m = np.ones(1000)
        x_m = c_m + 0.3 * a_m - 0.08 * a_m ** 2 + f_x.abduct_exogenous(observed_samples_dict)
        y_m = 0.2 * x_m - 1.0 * c_m + 0.1 * x_m * c_m - 0.2 * a_m + f_y.abduct_exogenous(observed_samples_dict)
        manual_counterfactuals = pd.DataFrame(
            data=np.column_stack((c_m, a_m, x_m, y_m)),
            columns=['c', 'a', 'x', 'y']
        )
        # compare manually created counterfactuals with computed counterfactuals
        self.assertTrue(np.abs(manual_counterfactuals - computed_counterfactuals).max().max() < 1e-10)

    def test_compute_counterfactuals_correct_model(self):
        """Test computation of counterfactuals for a StructuralCausalModel object.

        Here the model for the scm is the same as that used to generate the observed data.
        """
        # generate observed data by defining an scm and drawing sample
        f_c = nm.NodeAdditiveNoiseModel('c', {'a': [1.5]})
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]}, scale=5.)
        scm = cm.StructuralCausalModel(node_models=[f_c, f_a, f_x, f_y])
        observed_samples = scm.draw_sample(size=1000, initial_random_state=0)

        # compute counterfactuals for random values of c given observed data under this scm
        computed_counterfactuals = scm.compute_counterfactuals(
            observed_samples,
            intervention_variable='c',
            intervention_values=stats.norm.rvs(size=1000, random_state=50)
        )

        # manually create counterfactuals given the observed data under the scm
        # because the counterfactuals are calculated under an scm that was also used to generate the observed sample,
        # the abducted data for each non treatment node is equal to the exogenous noise when generating the observed
        # sample.
        a_m = stats.norm.rvs(size=1000, random_state=0)
        c_m = stats.norm.rvs(size=1000, random_state=50)
        x_m = c_m + 0.3 * a_m - 0.08 * a_m ** 2 + stats.norm.rvs(size=1000, random_state=2)
        y_m = 0.2 * x_m - 1.0 * c_m + 0.1 * x_m * c_m - 0.2 * a_m + stats.norm.rvs(size=1000, scale=5.,
                                                                                      random_state=3)
        manual_counterfactuals = pd.DataFrame(
            data=np.column_stack((c_m, a_m, x_m, y_m)),
            columns=['c', 'a', 'x', 'y']
        )
        # compare manually created counterfactuals with computed counterfactuals
        self.assertTrue(np.abs(manual_counterfactuals - computed_counterfactuals).max().max() < 1e-10)

    def test_compute_counterfactuals_df_index_incorrect_model(self):
        """Test dataframe index of output is correct for computation of counterfactuals for a StructuralCausalModel
        object.

        Here the observed data dataframe index is not in the default order and the model for the scm does not match that
        used to generate the observed data.
        """
        # generate observed data by defining an scm and drawing sample
        f_c_observed = nm.NodeAdditiveNoiseModel('c', {'a': [1.2]})
        f_a_observed = nm.NodeAdditiveNoiseModel('a')
        f_x_observed = nm.NodeAdditiveNoiseModel('x', {'c': [1.2], 'a': [0.35, -0.09]})
        f_y_observed = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'a': [-0.2]}, scale=5.)
        scm_observed = cm.StructuralCausalModel(node_models=[f_c_observed, f_a_observed, f_x_observed, f_y_observed])
        observed_samples = scm_observed.draw_sample(size=1000, initial_random_state=0)

        # define the scm model under which counterfactuals will be calculated
        # This scm model differs from the one used to generated observed data
        f_c = nm.NodeAdditiveNoiseModel('c', {'a': [1.5]})
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a': [0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]}, scale=5.)
        scm = cm.StructuralCausalModel(node_models=[f_c, f_a, f_x, f_y])

        # compute counterfactuals for random values of c given the observed data under this scm
        # we do this after shuffling the rows in the observed data and the counterfactual values of c
        rng = np.random.default_rng(seed=0)
        shuffled_index = rng.permutation(1000)

        computed_counterfactuals_shuffled_inputs = scm.compute_counterfactuals(
            observed_samples.loc[shuffled_index],
            intervention_variable='c',
            intervention_values=stats.norm.rvs(size=1000, random_state=50)[shuffled_index]
        )

        # calculate counterfactuals for original inputs
        computed_counterfactuals = scm.compute_counterfactuals(
            observed_samples,
            intervention_variable='c',
            intervention_values=stats.norm.rvs(size=1000, random_state=50)
        )

        # compare the counterfactual dataframes produced using original and shuffled inputs
        # the output dataframe that used shuffled inputs should be equal to the other after
        # aligning its index
        self.assertTrue(
            computed_counterfactuals_shuffled_inputs.loc[computed_counterfactuals.index].equals(
                computed_counterfactuals
            )
        )


if __name__ == "__main__":
    unittest.main()