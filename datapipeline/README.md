## 사용법 간단 설명

- workspace에서 `docker build -t your-image-name:your-tag -f datapipeline/Dockerfile .` 명령어를 실행한다.
- 그 후 `docker run --name my-running-app -it your-image-name:your-tag /bin/bash` 명령어로 컨테이너를 실행한다.