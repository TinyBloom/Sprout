# User Guide

Running the docker compose file compose-full.yaml to setup all the services:

```sh
docker-compose -f docker-compose/compose-full.yaml up
```

## API

Calling train API to train a new model:

```sh
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5001/api/ai/train
```

Response likes following:
```sh
{
  "model_id": "c4c8b049-e49a-438b-9497-3f6ba5a04d4f",
  "train_result": "success"
}
```

Using the model_id to detect anomalies in new data
```sh
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:5001/api/ai/predict -d '{"model_id": "c4c8b049-e49a-438b-9497-3f6ba5a04d4f", "training_data": [[1.2, 45.6, 0.8], [1.3, 46.2, 0.85], [1.1, 45.0, 0.75], [1.5, 47.0, 0.9]]}'
```

And the response is:
```sh
{
  "predict_result": [
    [
      1.2,
      45.6,
      0.8,
      68.00321591151024
    ],
    [
      1.3,
      46.2,
      0.85,
      37.488627086956626
    ],
    ...
  ]
}
```
