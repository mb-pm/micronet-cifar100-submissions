FROM tensorflow/tensorflow:1.14.0-gpu-py3

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

COPY run_pba.sh /workspace

COPY run_train_candidate_1.sh /workspace
COPY run_train_candidate_2.sh /workspace

COPY run_eval_candidate_1.sh /workspace
COPY run_eval_candidate_2.sh /workspace

