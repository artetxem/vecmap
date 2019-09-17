FROM tiangolo/python-machine-learning:latest

RUN git clone https://github.com/ngarneau/vecmap.git /vecmap
WORKDIR /vecmap

RUN pip install -r requirements.txt
CMD ["./reproduce.sh"]
