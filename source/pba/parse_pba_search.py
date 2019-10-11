import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import pba.augmentation_transforms_hp as augmentation_transforms_hp
from pba.data_utils import parse_policy


def __load_result_data(path):
    with open(path) as f:
        all_iterations_results = [json.loads(line) for line in f]
    iteration_to_results = defaultdict(list)
    for iteration_result in all_iterations_results:
        iteration_to_results[iteration_result['training_iteration']].append(iteration_result)
    return [
        sorted(
            iteration_to_results[result_data],
            key=lambda k: k['val_acc']
        )[-1] for result_data in sorted(iteration_to_results.keys())
    ]


def __parse_policy_hyperparams(full_policy):
    split = len(full_policy) // 2
    policy = parse_policy(
        full_policy[:split], augmentation_transforms_hp)
    policy.extend(parse_policy(
        full_policy[split:], augmentation_transforms_hp))
    return policy


def __convert(policy):
    name = policy[0]
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return (
        name,
        *policy[1:]
    )


if __name__ == '__main__':
    ray_result_dirs = glob.glob('/tmp/ray_results/cifar100_search/RayModel_*')
    ray_result_files = (os.path.join(d, 'result.json') for d in ray_result_dirs)

    results_data = tuple((__load_result_data(f) for f in ray_result_files))
    max_score_per_experiments = tuple(
        (max((iteration_data['val_acc'] for iteration_data in result_data)) for result_data in results_data)
    )
    best_result_idx = np.argmax(max_score_per_experiments)
    best_result_data = results_data[int(best_result_idx)]
    all_policies = (
        list((
            __convert(policy) for policy in __parse_policy_hyperparams(iteration_data['config']['hp_policy'])
        )) for iteration_data in best_result_data
    )
    for policy in all_policies:
        print(policy)
