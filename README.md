# Cross-Modal Pixel-and-Stroke Representation Aligning Networks for Free-Hand Sketch Recognition (CMPS)

This paper has been submitted to Expert Systems with Applications (ESWA).
<div align=center>
<img width="600" alt="1696749034040" src="https://github.com/WoodratTradeCo/CMPS/assets/38500652/aeb21671-edd6-489f-8571-43c029958842">
</div>
<div align=center>
<img width="700" alt="1696749476699" src="https://github.com/WoodratTradeCo/CMPS/assets/38500652/31b94d91-4b28-46d6-ac58-68b6fe5605de">
</div>

## Datasets
The code is based on Google QuickDraw-414K and TU-Berlin datasets. Thanks for the contributor, the source of QuickDraw-414K is from https://github.com/PengBoXiangShang/multigraph_transformer.
## Usage (How to Train Our CMPS)
The training log can be checked in experiment/log/CMPS_sota.log.

    # 1. Choose your workspace and download our repository.
    cd ${CUSTOMIZED_WORKSPACE}
    git clone https://github.com/PengBoXiangShang/multigraph_transformer
    # 2. Enter the directory.
    cd cmps
    # 3. Clone our environment, and activate it.
    conda-env create --name ${CUSTOMIZED_ENVIRONMENT_NAME}
    conda activate ${CUSTOMIZED_ENVIRONMENT_NAME}
    # 4. Download training/evaluation/testing dataset.
    # 5. Train our MGT. Please see details in our code annotations.
    # Please set the input arguments based on your case.
    # When the program starts running, a folder named 'experiment/${CUSTOMIZED_EXPERIMENT_NAME}' will be created automatically to save your log, checkpoint.
    python train.py 
    --exp ${CUSTOMIZED_EXPERIMENT_NAME}
    --epoch ${CUSTOMIZED_EPOCH}
    --batch_size ${CUSTOMIZED_SIZE}   
    --num_workers ${CUSTOMIZED_NUMBER} 
    --gpu ${CUSTOMIZED_GPU_NUMBER}
