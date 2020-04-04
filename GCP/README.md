# RL Model Training Setup on Google Cloud

We will be using Cloud Shell on Google Cloud Platform Console for all steps below

## 0. Pre-requisites
- Create a Google Cloud (GCP) Account
- GPU quota enabled in GCP
- Open Cloud Shell from GCP Console

## 1. Setup GCE VM
- This section is adapted from [fastai GCP setup](https://course.fast.ai/start_gcp.html)
- We want to use docker, CUDA 10.1 and conda package manager, leveraging on existing Google Cloud Deep Learning VM images
- Please ensure GPU quota is enabled, else please refer to fastai GCP setup link above, Step 3

```
export IMAGE_FAMILY="pytorch-latest-gpu"
export ZONE="us-west1-b"
export INSTANCE_NAME="drml"
export INSTANCE_TYPE="n1-highmem-4"

gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True"
```

## 2. Setup required packages, settings and code
- Ensure your project has been set in Cloud Shell, if not execute `gcloud config set project <project_id>`
- Login to VM from Cloud Shell `gcloud compute ssh --zone=us-west1-b jupyter@drml`
- Create new tmux session so that you can leave training running after closing cloud shell`tmux new-session -A -s airsimenv`
- Get project code `git clone https://github.com/raymondng76/IRS-Practice-Module-Dev.git`
- Create conda environment `sudo /opt/conda/bin/conda create -n airsim python=3.6.7`
- Activate conda environment `conda activate airsim`
- Install packages: `pip install -r IRS-Practice-Module-Dev/requirements.txt`
- Get AirSim: `git clone https://github.com/microsoft/AirSim.git`
- Update settings file
    - `rm AirSim/docker/settings.json`
    - `cp IRS-Practice-Module-Dev/airsim\ settings/settings.json.nodisplay AirSim/docker/`
    - `mv AirSim/docker/settings.json.nodisplay AirSim/docker/settings.json`

## 3. Build and run AirSim docker
- `cd AirSim/docker`
- Execute Build Script, targeting Ubuntu18.04 and CUDA 10.1:
```
python build_airsim_image.py \
   --base_image=nvidia/cudagl:10.1-devel-ubuntu18.04 \
   --target_image=airsim_binary:10.1-devel-ubuntu18.04
```
- Verify docker image built: `docker images | grep airsim`
- Get packaged Neighborhood AirSim Unreal Environment: `wget https://github.com/microsoft/AirSim/releases/download/v1.2.0Linux/Neighborhood.zip`
- Unzip to AirSim docker dir `unzip Neighborhood.zip -d .`
- Run environment in headless mode: `./run_airsim_image_binary.sh airsim_binary:10.1-devel-ubuntu18.04 Neighborhood/AirSimNH.sh -windowed -ResX=1080 -ResY=720 -- headless`
- Note: in `settings.json` file, no-display mode has also been setup
- Detach tmux session: `ctrl-b ctrl-b d`

## 4. Run model training file
- Create new session named code: `tmux new-session -A -s code`
- Activate conda environment `conda activate airsim`
- Download and copy yolo model weights to IRS-Practice-Module-Dev
    - `unzip yolo_drone_weights.zip -d IRS-Practice-Module-Dev/`
    - `cd IRS-Practice-Module-Dev`
    - `mv Yolov3\ Drone\ Weights/ weights`
- (Optional) If you are continuing training copy existing RL model weights to IRS-Practice-Module-Dev/code
    - `cd existing_results`
    - `cp -r . ../IRS-Practice-Module-Dev/code`
    - `cd ..`
- Execute required model training file in IRS-Practice-Module-Dev/code folder
    - `cd code`
    - `python <model>.py`
- Detach tmux session: `ctrl-b ctrl-b d`
- You can close cloud shell and let training to continue

## 5. Login to view progress
- Login to VM with user as jupyter from Cloud Shell `gcloud compute ssh --zone=us-west1-b jupyter@drml`
- View code progress or airsim output by typing tmux attach -t <SESSION> where SESSION can be `airsimenv` or `code`

## 6. Download RL model weights
- Install croc ([Github Ref](https://github.com/schollz/croc)): `curl https://getcroc.schollz.com | bash`
- From VM home folder, execute `croc send IRS-Practice-Module-Dev/code/save_model/<model>.h5` for required model
- Note the passcode output from previous line and execute command on local: `croc -yes <passcode>`
