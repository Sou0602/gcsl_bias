import doodad as dd
import gcsl.doodad_utils as dd_utils
import argparse

def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True, seed=0, **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gcsl import envs
    from gcsl.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gcsl.algo import buffer, variants_n11_off, networks_n11_off,gcsl_n11_sto_offline

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants_n11_off.get_params(env, env_params)

    if env_name == 'pointmass_empty':
        if seed == 0:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_empty/gcsl_o_1/2022_01_09_12_17_47/policy.pkl'
        elif seed == 1:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_empty/gcsl_o_1/2022_01_09_14_48_13/policy.pkl'
        elif seed == 2:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_empty/gcsl_o_1/2022_01_09_17_21_24/policy.pkl'
    elif env_name == 'pointmass_rooms':
        if seed == 0:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_rooms/gcsl_o_1/2022_01_09_12_17_45/policy.pkl'
        elif seed == 1:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_rooms/gcsl_o_1/2022_01_09_14_48_13/policy.pkl'
        elif seed == 2:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pointmass_rooms/gcsl_o_1/2022_01_09_17_21_24/policy.pkl'
    elif env_name == 'lunar':
        if seed == 0:
            model_path = '/home/nsh1609/gcsl-norm/data/example/lunar/gcsl_o_1/2022_01_09_12_17_47/policy.pkl'
        elif seed == 1:
            model_path = '/home/nsh1609/gcsl-norm/data/example/lunar/gcsl_o_1/2022_01_09_14_48_13/policy.pkl'
        elif seed == 2:
            model_path = '/home/nsh1609/gcsl-norm/data/example/lunar/gcsl_o_1/2022_01_09_17_21_24/policy.pkl'
    elif env_name == 'pusher':
        if seed == 0:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pusher/gcsl_o_1/2022_01_09_12_17_55/policy.pkl'
        elif seed == 1:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pusher/gcsl_o_1/2022_01_09_14_48_13/policy.pkl'
        elif seed == 2:
            model_path = '/home/nsh1609/gcsl-norm/data/example/pusher/gcsl_o_1/2022_01_09_17_21_24/policy.pkl'
    elif env_name == 'door':
        if seed == 0:
            model_path = '/home/nsh1609/gcsl-norm/data/example/door/gcsl_o_1/2022_01_09_12_17_55/policy.pkl'
        elif seed == 1:
            model_path = '/home/nsh1609/gcsl-norm/data/example/door/gcsl_o_1/2022_01_09_14_48_14/policy.pkl'
        elif seed == 2:
            model_path = '/home/nsh1609/gcsl-norm/data/example/door/gcsl_o_1/2022_01_09_17_21_25/policy.pkl'


    '''''
    algo = gcsl.GCSL(
        env,
        policy,
        replay_buffer,
        **gcsl_kwargs
    )
    '''''
    algo = gcsl_n11_sto_offline.GCSL(
        env,
        policy,
        replay_buffer,
        model_path,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/gcsl_off_n11_sto/' % (env_name,)

    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--seed", default='0')
    parser.add_argument("-E", "--env", default='pointmass_empty')

    args = parser.parse_args()
    seed = int(args.seed)
    env = args.env
    params = {
        'seed': [seed],
        'env_name':[env], #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': [True],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
