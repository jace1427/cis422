FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python3", "test_tree_and_pipelines.py"]
