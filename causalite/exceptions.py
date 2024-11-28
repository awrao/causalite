"""Custom exceptions."""

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


class MissingParentDataError(Exception):
    """Custom exception for when node cannot be sampled/abducted/predicted due to missing parent data"""
    pass


class MissingDataError(Exception):
    """Custom exception for when node cannot be abducted/predicted due to missing data"""
    pass


class MissingNodeError(Exception):
    """ Custom exception for when node is missing from a DAG"""
    pass
