# Codes are adopted from the original implementation
# https://github.com/vzantedeschi/DAGuerreotype/blob/main/daguerreo/run_model.py
# under the BSD 3-Clause License
# Copyright (c) 2023, Valentina Zantedeschi
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from daguerreo.models import Daguerro
from daguerreo import utils
from daguerreo.args import parse_pipeline_args
import networkx as nx
import typing as th

from base import AbstractBaseline


class Permutohedron(AbstractBaseline):
    """DAG Learning on the Permutohedron baseline. https://arxiv.org/pdf/2301.11898.pdf"""

    def __init__(
            self,
            dataset: th.Union["OCDDataset", str],  # type: ignore
            dataset_args: th.Optional[th.Dict[str, th.Any]] = None,
            # hyperparameters
            linear: bool = False,
            seed: int = 42,
    ):
        super().__init__(dataset=dataset, dataset_args=dataset_args, name='Permutohedron')
        self.linear = linear

        # parse args
        arg_parser = parse_pipeline_args()
        self.args = arg_parser.parse_args()
        self.seed = seed
        self.args.standardize = True
        self.args.sparsifier = 'none'
        self.args.equations = 'linear' if self.linear else 'nonlinear'

    def estimate_order(self):
        utils.init_seeds(seed=self.seed)
        samples = self.get_data(conversion="pandas")
        daguerro = Daguerro.initialize(samples, self.args, self.args.joint)
        daguerro, samples = utils.maybe_gpu(self.args, daguerro, samples)
        _ = daguerro(samples, utils.AVAILABLE[self.args.loss], self.args)
        daguerro.eval()
        _, dags = daguerro(samples, utils.AVAILABLE[self.args.loss], self.args)

        estimated_adj = dags[0].detach().cpu().numpy()
        g = nx.DiGraph(estimated_adj)
        orders = list(nx.topological_sort(g))
        return orders
