FROM jjanzic/docker-python3-opencv
COPY . /web
WORKDIR /web/api
RUN pip install Flask
RUN pip install Pillow==2.6.0
ENTRYPOINT ["python"]
CMD ["app.py"]