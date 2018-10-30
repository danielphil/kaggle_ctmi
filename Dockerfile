FROM python:3.5.6

MAINTAINER Ian Poole

COPY requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

COPY kaggle_ctmi /root/kaggle_ctmi
COPY tests /root/tests
COPY ShaipUnittestWorkspace/ /root/ShaipUnittestWorkspace/
COPY ShaipWorkspace/inputs/data /root/ShaipWorkspace/inputs/data
COPY ShaipWorkspace/inputs/groundtruth /root/ShaipWorkspace/inputs/groundtruth
COPY ShaipWorkspace/outputs/results/ /root/ShaipWorkspace/outputs/results/

WORKDIR /root/

ENV PYTHONPATH /root/
CMD pytest; python kaggle_ctmi/experiment.py


