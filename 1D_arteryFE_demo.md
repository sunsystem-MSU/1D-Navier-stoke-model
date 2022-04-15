## Run code via docker

# 1st git clone the folder run code:
'(sudo) git clone https://github.com/KVSlab/bloodflow.git'

# 2nd, in the folder directory Build the Docker image run code: 
'(sudo) docker build --no-cache -t arteryfe:2017.2.0 .'

# 3rd, create and enter a Docker container run code:
'docker run -ti -p 127.0.0.1:8000:8000 -v $(pwd):/home/fenics/shared -w /home/fenics/shared "arteryfe:2017.2.0"'

# The demo can be run using:
'python3 demo_arterybranch.py --cfg config/demo_arterybranch.cfg'
