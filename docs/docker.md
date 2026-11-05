# Run the prediction API in Docker

Start Docker Desktop, then run these commands in Git CMD from the project root:

```cmd
scripts\run-docker.cmd
curl.exe http://127.0.0.1:8001/health
curl.exe -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" --data-binary "@docs/examples/student_request.json"
```

The script builds `student-performance-api:local` from the Dockerfile and
starts a container named `student-performance-api`. It binds container port
8000 to `127.0.0.1:8001` on your computer. The API schema is at
`http://127.0.0.1:8001/docs`. The pipeline is copied into the image at build
time, so rebuild the image after changing the checked-in model file.

To inspect and follow logs:

```cmd
docker logs --tail 50 student-performance-api
docker logs --follow student-performance-api
```

Press `Ctrl+C` to stop following logs; the container keeps running. Stop it
with `docker stop student-performance-api`. The `--rm` flag removes the stopped
container, so the script can be run again.

If the script says the daemon is unavailable, wait until Docker Desktop is
fully started and `docker info` succeeds. If port 8001 is occupied, stop the
other service or change `8001` in `scripts/run-docker.cmd`. If the container
exits immediately, run the image in the foreground to see startup errors:

```cmd
docker run --rm --publish 127.0.0.1:8001:8000 student-performance-api:local
```

A missing or incompatible model prevents API startup.
