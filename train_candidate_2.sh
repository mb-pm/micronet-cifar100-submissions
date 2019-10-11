#!/bin/bash
if [[ "$1" == "--help" || "$1" == "--h" || "$#" -ne 1 ]]; then
    echo "Trains the model."
    echo "Arguments:"
    echo "  -use_best_augmentations: 1 if you want to use augmentations trained from our side, 0 if you want to use newly trained augmentations"
    exit 1
fi

USE_BEST_AUGMENTATIONS="$1"

full_source_path=$PWD/source

pushd environment
    docker build -t train_image train_and_eval
    docker run --rm -v ${full_source_path}:/workspace/source \
            -e ENVIRONMENT=DOCKER \
            train_image \
            bash /workspace/run_train_candidate_2.sh ${USE_BEST_AUGMENTATIONS}
popd