FROM python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip3 install -r requirements.txt
COPY main-web-socket.py .
COPY GA_Newest.py .
COPY parcels.py .
#COPY ./Algorithms/ .
EXPOSE 9696
#ENTRYPOINT [ "uvicorn" ]
#COPY ./algorithms /main-web-socket
#CMD [ "uvicorn", "main-web-socket:app", "--host", "0.0.0.0", "--port", "9696" ]