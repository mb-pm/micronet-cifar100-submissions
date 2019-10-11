"""Run PBA Search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import random
import numpy as np
import ray
from ray.tune import run_experiments
from ray.tune.schedulers import PopulationBasedTraining
import tensorflow as tf
from pba.train import RayModel
from pba.augmentation_transforms_hp import NUM_HP_TRANSFORM


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        default='simpnet',
    )
    parser.add_argument(
        '--data_path',
        default='/tmp/datasets/cifar-100-python/',
        help='Directory where dataset is located.')
    parser.add_argument(
        '--dataset',
        default='cifar100',
    )
    parser.add_argument(
        '--recompute_dset_stats',
        default=True,
        help='Instead of using hardcoded mean/std, recompute from dataset.'
    )
    parser.add_argument('--local_dir', type=str, default='/tmp/ray_results/', help='Ray directory.')
    parser.add_argument('--restore', type=str, default=None, help='If specified, tries to restore from given path.')
    parser.add_argument('--train_size', type=int, default=5000, help='Number of training examples.')
    parser.add_argument('--val_size', type=int, default=45000, help='Number of validation examples.')
    parser.add_argument('--checkpoint_freq', type=int, default=0, help='Checkpoint frequency.')
    parser.add_argument(
        '--cpu', type=float, default=40, help='Allocated by Ray')
    parser.add_argument(
        '--gpu', type=float, default=1, help='Allocated by Ray')
    parser.add_argument(
        '--aug_policy',
        type=str,
        default='cifar100',
        help=
        'which augmentation policy to use (in augmentation_transforms_hp.py)'
    )
    # search-use only
    parser.add_argument(
        '--explore',
        type=str,
        default='cifar100',
        help='which explore function to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='Number of epochs, or <=0 for default'
    )
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, help='weight decay')
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--test_bs', type=int, default=20, help='test batch size')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of Ray samples')
    parser.add_argument('--perturbation_interval', type=int, default=3)
    parser.add_argument('--name', type=str, default='cifar100_search')
    FLAGS = parser.parse_args()
    tf.logging.info('data path: {}'.format(FLAGS.data_path))
    hparams = tf.contrib.training.HParams(
        train_size=FLAGS.train_size,
        validation_size=FLAGS.val_size,
        dataset=FLAGS.dataset,
        data_path=FLAGS.data_path,
        batch_size=FLAGS.bs,
        gradient_clipping_by_global_norm=5.0,
        explore=FLAGS.explore,
        aug_policy=FLAGS.aug_policy,
        no_cutout=False,
        recompute_dset_stats=FLAGS.recompute_dset_stats,
        lr=FLAGS.lr,
        weight_decay_rate=FLAGS.wd,
        test_batch_size=FLAGS.test_bs)
    hparams.add_hparam('no_aug', False)
    hparams.add_hparam('use_hp_policy', True)
    # default start value of 0
    hparams.add_hparam('hp_policy', [0 for _ in range(4 * NUM_HP_TRANSFORM)])
    hparams.add_hparam('model_name', 'simpnet')
    hparams.add_hparam('num_epochs', FLAGS.epochs)
    hparams.set_hparam('batch_size', 128)
    tf.logging.info('epochs: {}, lr: {}, wd: {}'.format(hparams.num_epochs, hparams.lr, hparams.weight_decay_rate))
    hparams_config = hparams.values()

    train_spec = {
        "run": RayModel,
        "resources_per_trial": {
            "cpu": FLAGS.cpu,
            "gpu": FLAGS.gpu
        },
        "stop": {
            "training_iteration": hparams.num_epochs,
        },
        "config": hparams_config,
        "local_dir": FLAGS.local_dir,
        "checkpoint_freq": FLAGS.checkpoint_freq,
        "num_samples": FLAGS.num_samples,
    }

    if FLAGS.restore:
        train_spec["restore"] = FLAGS.restore

    def explore(config):
        """Custom explore function.

    Args:
      config: dictionary containing ray config params.

    Returns:
      Copy of config with modified augmentation policy.
    """
        new_params = []
        if config["explore"] == "cifar100":
            for i, param in enumerate(config["hp_policy"]):
                if random.random() < 0.2:
                    if i % 2 == 0:
                        new_params.append(random.randint(0, 10))
                    else:
                        new_params.append(random.randint(0, 9))
                else:
                    amt = np.random.choice(
                        [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
                    # Cast np.int64 to int for py3 json
                    amt = int(amt)
                    if random.random() < 0.5:
                        new_params.append(max(0, param - amt))
                    else:
                        if i % 2 == 0:
                            new_params.append(min(10, param + amt))
                        else:
                            new_params.append(min(9, param + amt))
        else:
            raise ValueError()
        config["hp_policy"] = new_params
        return config

    ray.init()

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="val_acc",
        perturbation_interval=FLAGS.perturbation_interval,
        custom_explore_fn=explore,
        log_config=True)

    run_experiments(
        {
            FLAGS.name: train_spec
        },
        scheduler=pbt,
        reuse_actors=True,
        verbose=True
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()