#!/bin/bash

full_source_path=$PWD/source

pushd environment
    docker build -t train_image train_and_eval
    docker run --rm -v ${full_source_path}:/workspace/source \
            -e ENVIRONMENT=DOCKER \
            train_image \
            bash /workspace/run_pba.sh
popd