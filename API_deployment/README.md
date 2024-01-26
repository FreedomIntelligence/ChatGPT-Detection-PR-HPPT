# API Deployment

Run the following to deploy the model on the server end, you should specify the host and port:
```
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > myapp.log 2>&1 &
```
