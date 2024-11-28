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
from scipy.special import expit

from causalite import node_models as nm
from causalite import causal_models as cm


class TestNodes(unittest.TestCase):

    def test_additive_gaussian_noise_draw_sample(self):
        """Test drawing sample for a NodeAdditiveNoiseModel object."""
        f_x = nm.NodeAdditiveNoiseModel('x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]})
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2. + stats.norm.rvs(size=sample_size, random_state=3)

        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_additive_gaussian_noise_pure_interaction_draw_sample(self):
        """Test drawing sample for a NodeAdditiveNoiseModel object with pure interaction model."""
        f_x = nm.NodeAdditiveNoiseModel('x', parent_polys={'ac': [-1.8, 3.5]})
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_2 = -1.8 * a_sample * c_sample + \
                     3.5 * (a_sample * c_sample) ** 2. + stats.norm.rvs(size=sample_size, random_state=3)

        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_additive_exponential_noise_draw_sample(self):
        """Test drawing sample for a NodeAdditiveNoiseModel object with exponential noise."""
        f_x = nm.NodeAdditiveNoiseModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]},
            u_draw_random_variates=stats.expon.rvs, loc=4., scale=5.
        )
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2. + stats.expon.rvs(
            size=sample_size, random_state=3, loc=4, scale=5.
        )

        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_additive_uniform_discrete_noise_draw_sample(self):
        """Test drawing sample for a NodeAdditiveNoiseModel object with uniform discrete noise."""
        f_x = nm.NodeAdditiveNoiseModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]},
            u_draw_random_variates=stats.randint.rvs, low=-1, high=2
        )
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2. + stats.randint.rvs(
            size=sample_size, random_state=3, low=-1, high=2
        )

        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_additive_custom_noise_draw_sample(self):
        """Test drawing sample for a NodeAdditiveNoiseModel object with custom noise."""

        # create custom rademacher distribution by creating an rv_discrete instance
        rademacher = stats.rv_discrete(name='rademacher', values=([-1, 1], [0.5, 0.5]))

        f_x = nm.NodeAdditiveNoiseModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]},
            u_draw_random_variates=rademacher.rvs
        )
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2. + rademacher.rvs(
            size=sample_size, random_state=3
        )

        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_binary_logistic_draw_sample_probs(self):
        """Test drawing sample probabilities for a NodeBinaryLogisticModel object."""
        f_x = nm.NodeBinaryLogisticModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]}
        )
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample_probs = f_x.draw_sample_probs(size=sample_size, parent_data=parent_data)
        x_sample_logits_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2.
        x_sample_probs_2 = expit(x_sample_logits_2)

        self.assertTrue(np.abs(x_sample_probs - x_sample_probs_2).max() < 1e-10)

    def test_binary_logistic_draw_sample(self):
        """Test drawing sample for a NodeBinaryLogisticModel object."""
        f_x = nm.NodeBinaryLogisticModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]}
        )
        sample_size = 5000
        a_sample = stats.norm.rvs(size=sample_size, random_state=1)
        c_sample = stats.norm.rvs(size=sample_size, random_state=2, loc=-1.)
        parent_data = {
            'a': a_sample,
            'c': c_sample
        }
        x_sample = f_x.draw_sample(size=sample_size, random_state=3, parent_data=parent_data)
        x_sample_latent_variable_2 = c_sample + 0.3 * a_sample - 0.08 * a_sample**2 - 1.8 * a_sample * c_sample +\
                     3.5 * (a_sample * c_sample) ** 2. + stats.logistic.rvs(size=sample_size, random_state=3)
        x_sample_2 = (x_sample_latent_variable_2 > 0.0) * 1.
        self.assertTrue(np.abs(x_sample - x_sample_2).max() < 1e-10)

    def test_abduct_exogenous_incorrect_model(self):
        """Test abduction of exogenous data for a NodeAdditiveNoiseModel object.

        Here the model for the node doesnt match the observed data model, as in the real world.
        """
        # create observed data according to 'true' model
        observed_size = 5000
        c_observed = stats.randint.rvs(low=-1, high=2, size=observed_size, random_state=0)
        a_observed = stats.norm.rvs(size=observed_size, random_state=1)
        x_observed = c_observed + 0.3 * a_observed - 0.08 * a_observed**2 + \
                     stats.norm.rvs(
                         size=observed_size, random_state=2
                     )
        y_observed = 0.2 * x_observed - 1. * c_observed - 0.2 * a_observed + \
                     stats.norm.rvs(scale=7.)

        # create node model for y
        # as in the real world, this doesn't match the 'true' underlying model for y_observed
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]})

        # abduct the exogenous data for the observed sample according to the node model for y
        y_abducted_exogenous = f_y.abduct_exogenous(
            {'c': c_observed, 'a': a_observed, 'x': x_observed, 'y': y_observed}
        )
        # manually determine the exogenous data according to the node model and test it is same as abducted
        y_abducted_exogenous_2 = y_observed - (
                0.2 * x_observed - 1. * c_observed + 0.1 * x_observed * c_observed - 0.2 * a_observed
        )
        self.assertTrue(np.abs(y_abducted_exogenous - y_abducted_exogenous_2).max() < 1e-10)

    def test_abduct_exogenous_correct_model(self):
        """Test abduction of exogenous node data for a NodeAdditiveNoiseModel object.

        Here the model for the node is the same as that used to generate the observed data.
        """
        # define node objects and generate observed data
        f_c = nm.NodeAdditiveNoiseModel('c', u_draw_random_variates=stats.randint.rvs, low=-1, high=2)
        f_a = nm.NodeAdditiveNoiseModel('a')
        f_x = nm.NodeAdditiveNoiseModel('x', {'c': [1.], 'a':[0.3, -0.08]})
        f_y = nm.NodeAdditiveNoiseModel('y', {'x': [0.2], 'c': [-1.], 'xc': [0.1], 'a': [-0.2]}, scale=5.)
        scm = cm.StructuralCausalModel(node_models=[f_c, f_a, f_x, f_y])
        observed_samples = scm.draw_sample(size=1000, initial_random_state=10)

        # abduct the exogenous data for node y using the generating model and the observed sample
        y_abducted_exogenous = f_y.abduct_exogenous(cm.samples_df_to_dict(observed_samples))
        # manually create the corresponding exogenous data according to the generating model and random state used to
        # draw the observed sample
        y_abducted_exogenous_2 = stats.norm.rvs(size=1000, scale=5., random_state=13)
        # test it is the same as the abducted exogenous data
        self.assertTrue(np.abs(y_abducted_exogenous - y_abducted_exogenous_2).max() < 1e-10)

    def test_additive_noise_model_predict(self):
        """Test prediction for a NodeAdditiveNoiseModel object."""
        f_x = nm.NodeAdditiveNoiseModel(
            'x', parent_polys={'c': [1.], 'a': [0.3, -0.08], 'ac': [-1.8, 3.5]},
            u_draw_random_variates=stats.expon.rvs, loc=0., scale=10.
        )
        predictions_size = 5000
        # create some predicted data for parents and another node
        a_prediction = stats.norm.rvs(size=predictions_size, random_state=1)
        c_prediction = stats.norm.rvs(size=predictions_size, random_state=2, loc=-1.)
        t_prediction = stats.norm.rvs(size=predictions_size, random_state=3, loc=6.7)
        predicted_parent_data = {
            'a': a_prediction,
            'c': c_prediction,
            't': t_prediction
        }
        # create some exogenous data for node x and those nodes
        a_exogenous = stats.uniform.rvs(size=predictions_size, loc=-3, scale=3.4)
        c_exogenous = stats.uniform.rvs(size=predictions_size, loc=20, scale=1.2)
        t_exogenous = stats.expon.rvs(size=predictions_size, loc=4., scale=5.)
        x_exogenous = stats.uniform.rvs(size=predictions_size, loc=-100, scale=10.)

        # predict for node x using method
        x_prediction = f_x.predict(
            predicted_parent_data=predicted_parent_data,
            abducted_data_exogenous={
                'a': a_exogenous, 'c': c_exogenous, 'x': x_exogenous, 't': t_exogenous
            }
        )
        # predict for node x manually
        x_prediction_2 = c_prediction + 0.3 * a_prediction - 0.08 * a_prediction**2 - 1.8 * a_prediction * c_prediction +\
                     3.5 * (a_prediction * c_prediction) ** 2. + x_exogenous

        # compare them
        self.assertTrue(np.abs(x_prediction - x_prediction_2).max() < 1e-10)


if __name__ == "__main__":
    unittest.main()