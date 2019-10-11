import argparse
import logging

import numpy as np
from tensorflow.python.keras.models import load_model

from counting import MicroNetCounter
from custom_layers import QuantBinaryConv
from utils import get_source_home_dir, load_data

_NUM_CLASSES = 100

logger = logging.getLogger('cnn')
logger.setLevel(logging.INFO)


def micronet_score(model):
    counter = MicroNetCounter(all_ops=model.layers, add_bits_base=32, mul_bits_base=32)
    (
        _, _, _, _, _,
        model_param_MB,
        op_mu,
        op_ad,
        model_op_MFLOPS
    ) = counter.print_summary(sparsity=0, param_bits=32, add_bits=32, mul_bits=32)

    def reformat_numbers(op_MFLOPS, par_MB):
        op_B = op_MFLOPS / 1e3
        par_M = par_MB / 4
        return op_B, par_M

    wrn_op_B = 10.49
    wrn_par_M = 36.5

    model_op_B, model_par_M = reformat_numbers(model_op_MFLOPS, model_param_MB)

    increase_ops = model_op_B / wrn_op_B
    increase_params = model_par_M / wrn_par_M
    print(f"Increase ops: {increase_ops} increase params: {increase_params}")
    print(f"Total Micronet score is {increase_ops+increase_params}")


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--use_best_model',
        help='Whether the best previously trained model should be loaded',
        type=int,
        default=1
    )

    args = argparser.parse_args()
    use_best_model = args.use_best_model == 1
    source_dir = get_source_home_dir() / 'outputs/quant_bin_simpnet'

    if use_best_model:
        logging.info('Using BEST model, which was trained from our side')
        full_path = source_dir / 'best_model'
    else:
        logging.info('Using freshly trained model.')
        full_path = source_dir / 'model'
        if not full_path.exists():
            print(source_dir)
            logging.error('There is no trained model. Please invoke training first')
            exit(1)

    best_model = load_model(
        full_path,
        custom_objects={
            'BinaryConv': QuantBinaryConv,
        }
    )

    (x_train, y_train), (x_test, y_test) = load_data()

    x_test = x_test.astype(np.float16)
    x_test = x_test.astype(np.float32)

    loss, acc = best_model.evaluate(
        x=x_test,
        y=y_test,
        batch_size=1000,
    )

    print("Accuracy is: {}".format(acc))
    micronet_score(best_model)
