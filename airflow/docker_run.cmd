#docker run -itd --name mlops-airflow -p 8080:8080 -e AIRFLOW_UID=$(id -u) mlops-airflow
docker run -itd \
  --name mlops-airflow \
  -p 8080:8080 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e AIRFLOW_UID=$(id -u) \
  mlops-airflow
