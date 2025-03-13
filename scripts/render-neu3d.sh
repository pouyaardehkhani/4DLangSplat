########## exp setup ##########
export centers_num=3
export ONLY_EVAL=t
clip_feat_dim=3
dataset_name=coffee_martini

########## time-agnostic language field ##########
export language_feature_hiddendim=${clip_feat_dim}
rm -rf submodules/4d-langsplat-rasterization/build 
pip install --no-cache-dir -e submodules/4d-langsplat-rasterization
export use_discrete_lang_f=f
for level in 1 2 3; do
for mode in "lang" "rgb"; do
python render.py -s  data/neu3d/${dataset_name} --model_path output/neu3d/${dataset_name}/${dataset_name}_${level} --skip_train --configs arguments/neu3d/${dataset_name}.py --mode ${mode} --no_dlang 1 --load_stage fine-lang 
done
done


