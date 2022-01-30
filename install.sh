pip install -r requirements.txt
pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.8.0/index.html

cd smartroom_ml/mmsegmentation || return 1
python setup.py install
cd ../..

cd smartroom_ml/XiaohuLuVPDetection || return 1
python setup.py build
python setup.py install
cd ../..

cd OneGan || return 1
python setup.py install
cd ..

mkdir -p smartroom_ml/neurvps_utils/logs
curl -L $(yadisk-direct https://disk.yandex.ru/d/Gf7_GFwu-iGoAA) -o smartroom_ml/neurvps_utils/logs/ScanNet.zip
unzip smartroom_ml/neurvps_utils/logs/ScanNet.zip -d smartroom_ml/neurvps_utils/logs/

mkdir -p smartroom_ml/mmsegmentation/checkpoints
curl -L https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth -o smartroom_ml/mmsegmentation/checkpoints/upernet_swin_base_patch4_window7_512x512.pth

mkdir -p smartroom_ml/lsun_room_master/ckpts
curl -L $(yadisk-direct https://disk.yandex.ru/d/68Rkv9aAYzUiPA) -o smartroom_ml/lsun_room_master/ckpts/model_retrained.ckpt

cd smartroom_ml/lama
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip
cd ../..

pip install -e .