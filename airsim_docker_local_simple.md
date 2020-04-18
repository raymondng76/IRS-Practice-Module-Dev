# Simplified instructions for running airsim environment with Docker locally

Modified from reference [Microsoft AirSim repository docker readme](https://github.com/microsoft/AirSim/blob/master/docs/docker_ubuntu.md)

## Building airsim docker container and environment

1. Either conda environment setup or virtualenv.
```
conda create -n airsim python=3.7
conda activate airsim
pip install airsim
```
2. Clone AirSim repo 
```
git clone https://github.com/microsoft/AirSim.git
```
3. Install nvidia-docker 2.0 per instructions [here](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))

4. Build image based on Ubuntu 18.04
```
python build_airsim_image.py \
   --base_image=nvidia/cudagl:10.1-devel-ubuntu18.04 \
   --target_image=airsim_binary:10.1-devel-ubuntu18.04
```
5. Verify image is now available
- `docker images | grep airsim`

## Running airsim environment using docker containers
- Go to directory `cd AirSim/docker` if you have not
- Replace the `settings.json` file with your desired configuration
- Execute `./download_blocks_env_binary.sh` to get default Blocks environment
- Execute `./run_airsim_image_binary.sh airsim_binary:10.1-devel-ubuntu18.04 Blocks/Blocks.sh -windowed -resX=1280 -resY=720` to run default Blocks environment
- Look for more environment in the [releases](https://github.com/microsoft/AirSim/releases) page
- For headless mode append ` -- headless` option. An example would be `./run_airsim_image_binary.sh airsim_binary:10.1-devel-ubuntu18.04 Blocks/Blocks.sh -windowed -resX=1280 -resY=720 -- headless`

