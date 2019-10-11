curl -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
mkdir -p /tmp/datasets
tar -C /tmp/datasets/ -xzf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz
