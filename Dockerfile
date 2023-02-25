FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8888

ENTRYPOINT [ "jupyter", "notebook", "--ip=0.0.0.0", "--notebook-dir=/tf", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''" ]