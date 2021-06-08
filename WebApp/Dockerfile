FROM python:3.6-slim
WORKDIR /webapp
COPY requirements.txt /webapp/
RUN pip install -r requirements.txt
COPY ./django-memcached-fix.py /usr/local/lib/python3.6/site-packages/django/core/cache/backends/memcached.py