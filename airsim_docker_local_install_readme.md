# Running airsim environment with Docker locally on Ubuntu 18.04 Readme

## Building airsim docker container and environment

1. Clone AirSim repo 
```
git clone https://github.com/microsoft/AirSim.git
```
2. Install nvidia-docker 2.0 per instructions [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

3. Build image based on Ubuntu 18.04
```
python build_airsim_image.py \
   --base_image=nvidia/cudagl:10.1-devel-ubuntu18.04 \
   --target_image=airsim_binary:10.1-devel-ubuntu18.04
```
4. Verify image is now available
- `docker images | grep airsim`

## Running airsim environment using docker containers
1. Go to directory `cd AirSim/docker` if you have not
2. Replace the `settings.json` file with your desired configuration from `airsim settings` folder
3. Execute `./download_blocks_env_binary.sh` to get default Blocks environment
4. Execute `./run_airsim_image_binary.sh airsim_binary:10.1-devel-ubuntu18.04 Blocks/Blocks.sh -windowed -resX=1280 -resY=720` to run default Blocks environment
5. Look for more environment in the [Microsoft AirSim Linux 1.2.0 release page](https://github.com/microsoft/AirSim/releases)
6. For headless mode append ` -- headless` option. An example would be `./run_airsim_image_binary.sh airsim_binary:10.1-devel-ubuntu18.04 Blocks/Blocks.sh -windowed -resX=1280 -resY=720 -- headless`


## Acknowledgements and References
Modified from reference [Microsoft AirSim repository docker readme](https://github.com/microsoft/AirSim/blob/master/docs/docker_ubuntu.md)

