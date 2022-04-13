# 2022 Max Planck institute for dynamics of complex technical systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from numpy import all
from typing import List, Dict, Tuple, Union, Set, FrozenSet
from mcs.parse_constr import *
from mcs.names import *
from optlang.interface import OPTIMAL, INFEASIBLE, UNBOUNDED

class SD_Solution(object):
    def __init__(self, model, sd, status, sd_setup):
        self.status = status
        self.sd_setup = sd_setup
        if GKOCOST in sd_setup or GKICOST in sd_setup:
            self.gene_sd = sd
            self.is_gene_sd = True
            # add here the computation of reaction strain designs from gene strain designs
        else:
            self.reaction_sd = sd
            self.is_gene_sd = False

    def get_sd(self):
        if self.is_gene_sd:
            return self.gene_sd
        else:
            return self.reaction_sd
    
    def get_rsd(self):
        return self.reaction_sd
    
    def get_gsd(self):
        return self.gene_sd
     
    def get_rsd_mark_ko_ki(self):
        pass
    
    def get_gsd_mark_ko_ki(self):
        pass