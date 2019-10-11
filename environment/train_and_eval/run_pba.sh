#!/bin/bash

bash /workspace/source/pba/download_cifar_100.sh
pushd /workspace/source/
    python3 -m pba.search
    python3 -m pba.parse_pba_search > /workspace/source/pba_augmentations.txt
popd