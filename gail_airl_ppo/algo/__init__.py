from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .rgcl import RGCL
from .avg import AVG
from .gail_avg import GAIL_AVG
from .airl_avg import AIRL_AVG
from .rgcl_avg import RGCL_AVG

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'rgcl': RGCL,
    'avg': AVG,
    'gail_avg': GAIL_AVG,
    'airl_avg': AIRL_AVG,
    'rgcl_avg': RGCL_AVG,
}
