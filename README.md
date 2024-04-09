**Docker Setup**

This project supports Docker to streamline setup, ensuring a consistent environment for development and testing. Follow the steps below to get started with Docker.

**Prerequisites**

Docker installed on your machine. Visit Docker's official site for installation instructions.

**Building the Docker Image:**

1. Clone the repository to your local machine:

`git clone https://github.com/yourusername/monkey_ID.git`

`cd monkey_ID`

2. Build the Docker image using the following command:

`docker build -t monkey_id .`

This command builds a Docker image named monkey_id based on the Dockerfile in the current directory.

**Running the Application in a Docker Container**

After building the image, you can run your application inside a Docker container:

`docker run -p 4000:8000 monkey_id`

This command starts a container from the monkey_id image, mapping port 8000 inside the container to port 4000 on your local machine. Adjust the port numbers as required by your application.

**Accessing the Application**

TBD

**Stopping the Container**

To stop the Docker container, you can press CTRL+C in the terminal where the container is running. Alternatively, you can stop the container by finding its container ID with docker ps and then using docker stop <monkey_id>.

Additional Docker Commands

Viewing running containers: docker ps

Listing all Docker images: docker images

For more advanced Docker functionalities, refer to the official Docker documentation.

