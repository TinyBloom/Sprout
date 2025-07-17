# Sprout

## Development

Setup virtual environment and install the requirements.

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Apply migrations locally
```
flask db upgrade
```

Setup flask app and celery worker/beat
```sh
python3 app.py
celery -A sprout.celery_worker.celery worker --loglevel=info
celery -A sprout.celery_worker.celery beat --loglevel=info
```

Run the pgsql locally
```
docker-compose -f compose-pgsql.yaml up -d
docker-compose -f compose-pgsql.yaml down -v
```
Run the mini locally
```
docker-compose -f compose-minio.yaml up -d
docker-compose -f compose-minio.yaml down -v
```

## User guide
[User Guide](doc/USER_GUIDE.md)

Swagger UI is available at /docs, for example you can open http://127.0.0.1:5001/docs and try REST APIs there.
