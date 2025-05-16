# FrenetTrajactoryTranslation
conda create -n ftt python=3.9.19 numpy=1.23.0
conda install pytorch=1.12.1 torchvision=0.13.1 cudatoolkit=11.3 -c pytorch

mkdir pyg_depend && cd pyg_depend
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_cluster-1.6.0%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.1.0%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_sparse-0.6.16%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_spline_conv-1.2.1%2Bpt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_cluster-1.6.0+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_scatter-2.1.0+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_sparse-0.6.16+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_spline_conv-1.2.1+pt112cu113-cp39-cp39-linux_x86_64.whl
python3 -m pip install torch_geometric

pip install scikit-learn networkx 
pip install iso3166
pip install shapely==1.7.1
pip install tensorboard
pip install pytorch-lightning