dest_path=$1
mkdir -p $dest_path
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/coffee_martini.zip -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cook_spinach.zip -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/cut_roasted_beef.zip -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z01 -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z02 -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.z03 -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_salmon_1_split.zip -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip -P ${dest_path}
wget https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/sear_steak.zip -P ${dest_path}

unzip ${dest_path}/coffee_martini.zip -d ${dest_path}
unzip ${dest_path}/cook_spinach.zip -d ${dest_path}
unzip ${dest_path}/cut_roasted_beef.zip -d ${dest_path}
unzip ${dest_path}/flame_steak.zip -d ${dest_path}
unzip ${dest_path}/sear_steak.zip -d ${dest_path}
zip -F ${dest_path}/flame_salmon_1_split.zip --out ${dest_path}/flame_salmon_1.zip
unzip ${dest_path}/flame_salmon_1.zip -d ${dest_path}

rm ${dest_path}/*.zip
rm ${dest_path}/flame_salmon_1_split.z0*

for dataset_name in "coffee_martini" "cook_spinach" "cut_roasted_beef" "flame_salmon_1" "flame_steak" "sear_steak"; do
    echo "Dataset:${dataset_name} Extrace images from videos."
    python  preprocess/preprocess_neu3d.py --datadir ${dest_path}/${dataset_name}
done

echo "Done."