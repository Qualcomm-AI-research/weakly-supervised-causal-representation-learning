FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR /app

RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install scikit-learn==0.24.2
RUN pip3 install hydra-core==1.3.1
RUN pip3 install mlflow==2.1.1
RUN pip3 install nflows==0.14 protobuf==3.20.1
RUN pip3 install seaborn==0.12.2

COPY . .

RUN pip install -e .
