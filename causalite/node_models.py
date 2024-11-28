"""Models for the causal mechanism of a variable in a causal model."""

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

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm, logistic
from scipy.special import expit

from .exceptions import MissingDataError, MissingParentDataError


class NodePostTransformedPolynomialModel(ABC):
    """Model of a specific form for the causal mechanism of a variable in an SCM/node in corresponding DAG.

    The model for the causal mechanism of a variable X is of form
    X = g(f(Pa(X)) + U_X)
    where Pa(X) are its parents, U_X is exogenous variation (noise), and g and f are functions. Here, f is a sum of
    functions q_i, where each q_i is a polynomial function of a single parent variable, or a polynomial function of the
    first order interaction between two parent variables. The post transformation function g is not defined within this
    abstract class, rather it is defined within a subclass through its `draw_sample` method.

    Parameters
    ----------
    name: string of length 1
        Name of variable/node.

    parent_polys: dict of str: list, optional
        Name of parent variable or first order pairwise
        interaction of parent variables, mapped to its polynomial function.
        The polynomial function q is a list such that
        q(variable) = sum list[i] * variable ** (i+1).

    u_draw_random_variates: method, default=scipy.stats.norm.rvs
        Method that draws random variates from the exogenous noise distribution. Expected to
        be an rvs method of an implemented or custom distribution from scipy.stats.

    **u_kwargs: Distribution parameters of `u_draw_random_variates`, optional

    Attributes
    ----------
    is_root: bool
        Whether the node is a root node (has no parents) or not.

    parent_variables: list of strings
        List of parent variables, sorted alphabetically.
    """

    def __init__(self, name, parent_polys=None, u_draw_random_variates=None, **u_kwargs):

        # validate node name
        self._validate_name(name)
        self.name = name
        self.parent_polys = {}
        self.is_root = True

        # store parent polynomials if provided and change is_root attribute
        if parent_polys is not None:
            # validate parent poly keys
            self._validate_parent_poly_keys(parent_polys)
            self.parent_polys = parent_polys
            self.is_root = False

        # parse exogenous distribution
        if u_draw_random_variates is None:
            u_draw_random_variates = norm.rvs
        self.u_draw_random_variates = u_draw_random_variates

        # store exogenous distribution parameters
        self.u_params = u_kwargs

        # get and store parent variables
        self.parent_variables = self.get_parent_variables()

    @staticmethod
    def _validate_name(node_name):
        """Check node name is valid."""
        if len(node_name) > 1:
            raise ValueError("node name must be a string of length 1")

    @staticmethod
    def _validate_parent_poly_keys(parent_polys):
        """Check parent poly keys are valid."""
        if any(len(k) > 2 for k in list(parent_polys.keys())):
            raise ValueError("parent poly keys must be strings of maximum length 2")

    def get_parent_variables(self):
        """Get list of parent variables sorted alphabetically."""
        parent_variables = set()
        for poly_argument in self.parent_polys.keys():
            if len(poly_argument) == 1:
                parent_variables.add(poly_argument)
            elif len(poly_argument) == 2:
                parent_variables.add(poly_argument[0])
                parent_variables.add(poly_argument[1])
        return list(np.sort(list(parent_variables)))

    @staticmethod
    def _evaluate_polynomial(sample, poly_coefficients):
        """Evaluate polynomial on a sample."""
        result = 0
        for (idx, coefficient) in enumerate(poly_coefficients, start=1):
            result += coefficient * pow(sample, idx)
        return result

    def _draw_sample_pre_transformed_exogenous(self, size=1000, random_state=0):
        """Draw pre-transformed exogenous part of sample of the variable.

        Parameters
        ----------
        size: int, default=1000
            Size of the sample
        random_state: int, default=0
            Random state used by self.u_draw_random_variates when drawing exogenous noise.

        Returns
        -------
        sample_pre_transformed_exogenous: numpy array
            Sample of size (`size`,).
        """
        # draw from exogenous part
        sample_pre_transformed_exogenous = self.u_draw_random_variates(**{
            **self.u_params,
            'size': size,
            'random_state': random_state
        })
        return sample_pre_transformed_exogenous

    def _draw_sample_pre_transformed_endogenous(self, size=1000, parent_data=None):
        """Draw pre-transformed endogenous part of sample of variable.

        Parameters
        ----------
        size: int, default=1000
            Size of the sample.
        parent_data: dict of str: numpy array, default=None
            Name of parent variable mapped to parent sample. Non-parent variables included
            in `parent_data` will be ignored.

        Returns
        -------
        sample_pre_transformed_endogenous: numpy array
            Sample of size (`size`,).
        """
        sample_pre_transformed_endogenous = np.zeros(size)
        # add polynomial functions and interactions of parent variables for
        # non-root nodes
        if self.is_root is False:
            # check that parent data is non-empty for non-root node
            if parent_data is None:
                raise MissingParentDataError(f"cannot determine data for variable {self.name} because no parent data supplied")
            # loop over all the parent variables
            for poly_argument, poly_coefficients in self.parent_polys.items():
                if poly_argument in self.parent_variables:
                    # evaluate non-interaction parent polynomials
                    #
                    if poly_argument in parent_data.keys():
                        sample_pre_transformed_endogenous += self._evaluate_polynomial(parent_data[poly_argument], poly_coefficients)
                    else:
                        raise MissingParentDataError(f"cannot determine data for variable {self.name} because no data for parent \
{poly_argument} was supplied")
                else:
                    # evaluate interaction parent polynomials
                    # we do this by multiplying the corresponding parent samples
                    # and passing this as the sample for evaluation

                    if poly_argument[0] in parent_data.keys() and poly_argument[1] in parent_data.keys():
                        sample_pre_transformed_endogenous += self._evaluate_polynomial(
                            parent_data[poly_argument[0]] * parent_data[poly_argument[1]],
                            poly_coefficients
                        )
                    else:
                        raise MissingParentDataError(f"cannot determine data for variable {self.name} because data missing for parent \
{poly_argument[0]} or {poly_argument[1]}")
        return sample_pre_transformed_endogenous

    def _draw_sample_pre_transformed(self, size=1000, random_state=0, parent_data=None):
        """Draw pre-transformed sample of the variable.

        Parameters
        ----------
        size: int, default=1000
            Size of the sample.
        random_state: int, default=0
            Random state used by self.u_draw_random_variates when drawing exogenous noise.
        parent_data: dict of str: numpy array, default=None
            Name of parent variable mapped to parent sample. Non-parent variables included
            in `parent_data` will be ignored.

        Returns
        -------
        sample_pre_transformed: numpy array
            Sample of size (`size`,).
        """
        sample_pre_transformed_exogenous = self._draw_sample_pre_transformed_exogenous(size=size, random_state=random_state)
        sample_pre_transformed_endogenous = self._draw_sample_pre_transformed_endogenous(size=size, parent_data=parent_data)
        sample_pre_transformed = sample_pre_transformed_exogenous + sample_pre_transformed_endogenous
        return sample_pre_transformed

    @abstractmethod
    def draw_sample(self, size=1000, random_state=0, parent_data=None):
        """Draw sample of the variable.

        Needs to be implemented in child class to take into account the specific post transformation function associated
        with the child class.

        Parameters
        ----------
        size: int, default=1000
            Size of the sample.
        random_state: int, default=0
            Random state used by self.u_draw_random_variates when drawing exogenous noise.
        parent_data: dict of str: numpy array, default=None
            Name of parent variable mapped to parent sample. Non-parent variables included
            in `parent_data` will be ignored.

        Returns
        -------
        sample: numpy array
            Sample of size (`size`,).
        """
        pass

    @staticmethod
    def _polynomial_to_string(poly_argument, poly_coefficients):
        """Create string representation of polynomial function."""
        result = ''
        for (idx, coefficient) in enumerate(poly_coefficients, start=1):
            if coefficient != 0.0:
                # append the coefficient and variable name
                # need brackets for interaction arguments of idx > 1
                if len(poly_argument) == 2 and idx > 1:
                    result += ' + ' + str(coefficient) + '(' + poly_argument + ')'
                else:
                    result += ' + ' + str(coefficient) + poly_argument
                # include exponent if greater than 1
                if idx > 1:
                    result += '^' + str(idx)

        # deal with any + -
        result = result.replace('+ -', '- ')
        # deal with any leading + or spaces
        result = result.strip(' +')
        return result

    @abstractmethod
    def __str__(self):
        """Create string representation of the model.

        Below assumes there is no post-transformation, so should be overidden or extended in child class according
        to transformation function.
        """
        # create exogenous part
        # u prefix is given same case as the name of variable
        if str.isupper(self.name):
            u_prefix = 'U_'
        else:
            u_prefix = 'u_'
        exogenous_part = u_prefix + self.name

        # non-root nodes
        if self.is_root is False:
            endogenous_part = ''
            for (poly_argument, poly_coefficients) in self.parent_polys.items():
                # append each parent polynomial string
                endogenous_part += ' + ' + self._polynomial_to_string(
                    poly_argument, poly_coefficients
                )

            # deal with any + -
            endogenous_part = endogenous_part.replace('+ -', '-')
            # replace any leading or trailing + or spaces
            endogenous_part = endogenous_part.strip(' +')
            # prepend variable name and append exogenous part
            model_as_string = self.name + ' <-  ' + endogenous_part + ' + ' + exogenous_part
        else:
            # root node
            # string is just variable name and exogenous part
            model_as_string = self.name + ' <-  ' + exogenous_part
        return model_as_string


class NodeAdditiveNoiseModel(NodePostTransformedPolynomialModel):
    """Additive Noise Model for the causal mechanism of a variable in an SCM/node in corresponding DAG.

    This is of form
    X = f(Pa(X)) + U_X
    See parent class for further details and parameters and attributes.
    """

    def draw_sample(self, *args, **kwargs):
        """Draw sample of the variable.

        See draw_sample method of parent class for specification.
        """

        # draw pre-transformed sample
        sample_pre_transformed = self._draw_sample_pre_transformed(*args, **kwargs)
        # no post-transformation required
        sample = sample_pre_transformed
        return sample

    def abduct_exogenous(self, observed_variable_and_parent_data):
        """Abduct exogenous data for variable given observed variable and parent data.

        Parameters
        ----------
        observed_variable_and_parent_data: dict of str: numpy array
            Name of variable or parent variable mapped to variable or parent sample. Other variables included
            in `observed_variable_and_parent_data` will be ignored.

        Returns
        -------
        abducted_variable_data_exogenous: numpy array
            array containing the exogenous data abducted from the observed data.
        """
        # check that observed data for variable was supplied
        if self.name not in observed_variable_and_parent_data.keys():
            raise MissingDataError(f"cannot abduct exogenous data for variable {self.name} as its observed data was not supplied")

        # get the size of observed sample
        size = observed_variable_and_parent_data[self.name].size

        # get the observed data for variable
        observed_variable_data = observed_variable_and_parent_data[self.name].copy()

        # determine pre transformed endogenous part of observed data for variable
        observed_variable_data_pre_transformed_endogenous = self._draw_sample_pre_transformed_endogenous(
            size=size, parent_data=observed_variable_and_parent_data
        )

        # apply inverse of post transformation to observed variable data to get pre-transformed data
        # here the inverse is identity
        observed_variable_data_pre_transformed = observed_variable_data
        # subtract the pre transformed endogenous part leaving the exogenous data
        abducted_variable_data_exogenous = observed_variable_data_pre_transformed - observed_variable_data_pre_transformed_endogenous
        return abducted_variable_data_exogenous

    def predict(self, predicted_parent_data, abducted_data_exogenous):
        """Predict data for variable given predicted parent data and abducted exogenous data.

        Parameters
        ----------
        predicted_parent_data: dict of str: numpy array
            Name of parent variable mapped to predicted parent data. Other variables included
            in `predicted_parent_data` will be ignored.

        abducted_data_exogenous: dict of str: numpy array
            Name of variable mapped to variable abducted exogenous data. Other variables included
            in `abducted_data_exogenous` will be ignored.

        Returns
        -------
        predicted_variable_data: numpy array
            array containing the predicted variable data.
        """
        # check that abducted exogenous data for this variable was supplied
        if self.name not in abducted_data_exogenous.keys():
            raise MissingDataError(f"cannot predict variable {self.name} as its abducted exogenous data was not supplied")

        # get the size of data for prediction
        size = abducted_data_exogenous[self.name].size
        # compute endogenous part
        predicted_variable_data_pre_transformed_endogenous = self._draw_sample_pre_transformed_endogenous(
            size=size, parent_data=predicted_parent_data
        )
        # add abducted exogenous part
        predicted_variable_data_pre_transformed = predicted_variable_data_pre_transformed_endogenous + \
                                              abducted_data_exogenous[self.name]

        # no post-transformation required
        predicted_variable_data = predicted_variable_data_pre_transformed
        return predicted_variable_data

    def __str__(self):
        """Create string representation of the model."""
        # get the string representation when there is no post-transformation and use that
        model_as_string = super().__str__()
        return model_as_string


class NodeBinaryLogisticModel(NodePostTransformedPolynomialModel):
    """Binary Logistic Model for the causal mechanism of a variable in an SCM/node in a corresponding DAG.

    This is of form
    X = I( f(Pa(X)) + U_X > 0 )
    where X is a binary variable taking values in {0, 1}, I is the indicator function, U_X ~ logistic(0,1).

    It is the latent-value formulation of the equivalent probabilistic model
    P(X = 1 | Pa(X)) = sigmoid(f(Pa(X))
    
    Parameters
    ----------
    name: See NodePostTransformedPolynomialModel
    parent_polys: See NodePostTransformedPolynomialModel

    Attributes
    ----------
    is_root: See NodePostTransformedPolynomialModel
    parent_variables: See NodePostTransformedPolynomialModel

    Notes
    ----------
    The equivalence between the latent-value and probabilistic formulation is taken from [1].

    References
    __________

    [1] 'Data Analysis Using Regression and Multilevel/Hierarchical Models', Andrew Gelman and Jennifer Hill, Cambridge
    University Press, 2007.
    """
    def __init__(self, name, parent_polys=None):
        # set the correct u_draw_random_variates and parameters and call parent initializer
        super().__init__(name, parent_polys, u_draw_random_variates=logistic.rvs)

    def draw_sample_probs(self, *args, **kwargs):
        """Draw random sample of the probabilities of the variable.

        This method essentially extends the draw_sample_pre_transformed_endogenous method of parent class.

        Parameters
        ----------
        *args: see draw_sample_pre_transformed_endogenous method of parent class
        **kwargs: see draw_sample_pre_transformed_endogenous method of parent class

        Returns
        -------
        sample_probs: numpy array
            Sample of probabilities of size (size,).
        """
        # draw logits of sample probs
        logit_sample_probs = self._draw_sample_pre_transformed_endogenous(*args, **kwargs)
        # transform to sample probs
        sample_probs = expit(logit_sample_probs)

        return sample_probs

    def draw_sample(self, *args, **kwargs):
        """Draw random sample of the variable. See draw_sample method of parent class for specification."""

        sample_latents = self._draw_sample_pre_transformed(*args, **kwargs)
        # apply thresholding
        sample = (sample_latents > 0.0) * 1.
        return sample
    
    def __str__(self):
        """Create string representation of the model. Extends method of parent class."""
        # get string representation without binarization
        model_as_string = super().__str__()

        # put post-transformation function into string
        model_as_string = model_as_string.replace(
            " <-  ", " <-  I( "
        )
        model_as_string += " > 0.0 )"
        return model_as_string



