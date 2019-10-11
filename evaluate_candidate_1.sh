#!/bin/bash

if [[ "$1" == "--help" || "$1" == "--h" || "$#" -ne 1 ]]; then
    echo "Runs evaluation on trained model.
    "
    echo "Arguments:"
    echo "  -use_best_model: 1 if you want to use best model, 0 if you want to use trained model"
    exit 1
fi

USE_BEST_MODEL="$1"

full_source_path=$PWD/source

pushd environment
    docker build -t train_image train_and_eval
    docker run --rm -v ${full_source_path}:/workspace/source \
            -e ENVIRONMENT=DOCKER \
            train_image \
            bash /workspace/run_eval_candidate_1.sh ${USE_BEST_MODEL}
popd